package lazyrnn

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/lazyseq"
)

// BPTT applies the block to the sequence using
// back-propagation through time.
//
// The sequence is produced lazily as the input is read.
// Thus, this is fundamentally different than doing
//
//     lazyseq.Lazify(anyrnn.Map(lazyseq.Unlazify(seq), block))
//
// which does not produce any output until the entire
// input sequence has been generated.
func BPTT(in lazyseq.Seq, block anyrnn.Block) lazyseq.Seq {
	return rnnFragmentToSeq(in, block, bptt(in.Forward(), block, nil))
}

func bptt(in <-chan *anyseq.Batch, block anyrnn.Block, start anyrnn.State) rnnFragment {
	outChan := make(chan *anyseq.Batch, 1)
	doneChan := make(chan struct{})
	frag := &bpttFrag{forward: outChan, done: doneChan, v: anydiff.VarSet{}}

	go func() {
		state := start
		for batch := range in {
			if state == nil {
				state = block.Start(len(batch.Present))
			}
			if batch.NumPresent() != state.Present().NumPresent() {
				state = state.Reduce(batch.Present)
			}
			res := block.Step(state, batch.Packed)
			frag.reses = append(frag.reses, res)
			state = res.State()
			outChan <- &anyseq.Batch{
				Packed:  res.Output(),
				Present: state.Present(),
			}
			frag.v = anydiff.MergeVarSets(frag.v, res.Vars())
		}
		close(outChan)
		close(doneChan)
	}()

	return frag
}

type bpttFrag struct {
	forward <-chan *anyseq.Batch

	// Fields are immutable once done is closed.
	done  <-chan struct{}
	reses []anyrnn.Res
	v     anydiff.VarSet
}

func (b *bpttFrag) Forward() <-chan *anyseq.Batch {
	return b.forward
}

func (b *bpttFrag) Vars() anydiff.VarSet {
	<-b.done
	return b.v
}

func (b *bpttFrag) Propagate(down chan<- *anyseq.Batch, up <-chan *anyseq.Batch,
	stateUp anyrnn.StateGrad, grad lazyseq.Grad) anyrnn.StateGrad {
	for _ = range b.forward {
	}

	nextGrad := stateUp
	for j := len(b.reses) - 1; j >= 0; j-- {
		res := b.reses[j]
		pres := res.State().Present()
		if nextGrad != nil && nextGrad.Present().NumPresent() != pres.NumPresent() {
			nextGrad = nextGrad.Expand(pres)
		}
		upBatch, ok := <-up
		if !ok {
			panic("not enough upstream batches")
		}
		upVec := upBatch.Packed
		var inDown anyvec.Vector
		grad.Use(func(g anydiff.Grad) {
			inDown, nextGrad = res.Propagate(upVec, nextGrad, g)
		})
		if down != nil {
			down <- &anyseq.Batch{
				Packed:  inDown,
				Present: upBatch.Present,
			}
		}
	}
	return nextGrad
}
