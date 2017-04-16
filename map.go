package lazyrnn

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anyvec"
)

// Map applies the function f to the timesteps of seq.
//
// When f is called, it is given the batch size and the
// packed batch vector.
func Map(seq Rereader, f func(n int, v anydiff.Res) anydiff.Res) Rereader {
	return MapN(func(n int, v ...anydiff.Res) anydiff.Res {
		return f(n, v[0])
	}, seq)
}

type mapNRes struct {
	Ins  []Rereader
	F    func(n int, v ...anydiff.Res) anydiff.Res
	Outs <-chan *anyseq.Batch

	Done <-chan struct{}
	Len  int
	V    anydiff.VarSet
}

// MapN applies a function to every timestep in a batch
// of Rereaders.
//
// The function f is called for a given timestep with the
// packed vectors from each sequence at that timestep.
// It is also supplied with the batch size.
//
// All of the sequences must be the same length, and each
// timestep must have the same present map.
//
// Due to the lazy-ness of the result, f may be called
// multiple times on the same timestep.
// Also, f may be called in any order.
//
// It is invalid to map over 0 sequences.
func MapN(f func(n int, v ...anydiff.Res) anydiff.Res, s ...Rereader) Rereader {
	if len(s) == 0 {
		panic("need at least one sequence")
	}
	out := make(chan *anyseq.Batch, 1)
	done := make(chan struct{})
	res := &mapNRes{
		Ins:  s,
		F:    f,
		Outs: out,
		Done: done,
		V:    anydiff.VarSet{},
	}
	go res.forward(out, done)
	return res
}

func (m *mapNRes) Creator() anyvec.Creator {
	return m.Ins[0].Creator()
}

func (m *mapNRes) Forward() <-chan *anyseq.Batch {
	return m.Outs
}

func (m *mapNRes) Vars() anydiff.VarSet {
	<-m.Done
	return m.V
}

func (m *mapNRes) Propagate(upstream <-chan *anyseq.Batch, grad *Grad) {
	for _ = range m.Forward() {
	}

	seqs := make([]Seq, len(m.Ins))
	for i, x := range m.Ins {
		seqs[i] = x
	}
	downstream, wg := propagateMany(seqs, grad)

	for idx := m.Len - 1; idx >= 0; idx-- {
		inChans := make([]<-chan *anyseq.Batch, len(m.Ins))
		for i, in := range m.Ins {
			inChans[i] = in.Reread(idx, idx+1)
		}
		u, ok := <-upstream
		if !ok {
			panic("not enough upstream batches")
		}
		down := m.propThroughF(inChans, u, grad)
		for i, downBatch := range down {
			if downstream[i] != nil {
				downstream[i] <- downBatch
			}
		}
	}

	if _, ok := <-upstream; ok {
		panic("too many upstream batches")
	}

	for _, ch := range downstream {
		if ch != nil {
			close(ch)
		}
	}

	wg.Wait()
}

func (m *mapNRes) Reread(start, end int) <-chan *anyseq.Batch {
	res := make(chan *anyseq.Batch, 1)
	chans := make([]<-chan *anyseq.Batch, len(m.Ins))
	for i, in := range m.Ins {
		chans[i] = in.Reread(start, end)
	}
	go func() {
		m.readAndApply(chans, res)
		close(res)
	}()
	return res
}

func (m *mapNRes) readAndApply(chans []<-chan *anyseq.Batch, out chan<- *anyseq.Batch) int {
	var count int
	for {
		var ins []anydiff.Res
		var present []bool
		var numPres int
		for _, ch := range chans {
			if in, ok := <-ch; ok {
				if len(ins) > 0 {
					if !presentMapsEqual(present, in.Present) {
						panic("present map mismatch")
					}
				}
				numPres = in.NumPresent()
				present = in.Present
				ins = append(ins, anydiff.NewConst(in.Packed))
			}
		}
		if len(ins) == 0 {
			break
		} else if len(ins) != len(chans) {
			panic("sequence length mismatch")
		}
		count++
		outVec := m.F(numPres, ins...).Output()
		out <- &anyseq.Batch{Packed: outVec, Present: present}
	}
	return count
}

func (m *mapNRes) forward(out chan<- *anyseq.Batch, done chan<- struct{}) {
	inChans := make([]<-chan *anyseq.Batch, len(m.Ins))
	for i, x := range m.Ins {
		inChans[i] = x.Forward()
	}
	m.Len = m.readAndApply(inChans, out)
	for _, in := range m.Ins {
		m.V = anydiff.MergeVarSets(m.V, in.Vars())
	}
	close(done)
	close(out)
}

// propThroughF calls m.F with the inputs, propagates
// through the result, and returns the downstream
// gradient.
func (m *mapNRes) propThroughF(ins []<-chan *anyseq.Batch, upstream *anyseq.Batch,
	grad *Grad) []*anyseq.Batch {
	var batchSize int
	var present []bool
	inReses := make([]anydiff.Res, len(m.Ins))
	inPools := make([]*anydiff.Var, len(m.Ins))
	for i, ch := range ins {
		batch := <-ch
		present = batch.Present
		batchSize = batch.NumPresent()
		inPools[i] = anydiff.NewVar(batch.Packed)
		inReses[i] = inPools[i]
	}

	var downstream []*anyseq.Batch
	grad.Use(func(g anydiff.Grad) {
		for _, pool := range inPools {
			g[pool] = pool.Vector.Creator().MakeVector(pool.Vector.Len())
		}
		out := m.F(batchSize, inReses...)
		out.Propagate(upstream.Packed, g)
		for _, pool := range inPools {
			downstream = append(downstream, &anyseq.Batch{
				Packed:  g[pool],
				Present: present,
			})
			delete(g, pool)
		}
	})

	return downstream
}

func presentMapsEqual(p1, p2 []bool) bool {
	if len(p1) != len(p2) {
		return false
	}
	for i, x := range p1 {
		if x != p2[i] {
			return false
		}
	}
	return true
}
