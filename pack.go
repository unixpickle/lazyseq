package lazyrnn

import (
	"sync"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/essentials"
)

type packSeqRes struct {
	C   anyvec.Creator
	Ins []Seq
	Out <-chan *anyseq.Batch

	Done        <-chan struct{}
	LanesPerSeq []int
	Lens        []int
	V           anydiff.VarSet
}

// PackSeq aggregates multiple Seqs together into a single
// Seq with larger batches.
func PackSeq(c anyvec.Creator, seqs []Seq) Seq {
	outChan := make(chan *anyseq.Batch, 1)
	doneChan := make(chan struct{})

	res := &packSeqRes{
		C:           c,
		Ins:         seqs,
		Out:         outChan,
		Done:        doneChan,
		LanesPerSeq: make([]int, len(seqs)),
		Lens:        make([]int, len(seqs)),
		V:           anydiff.VarSet{},
	}

	go res.forward(outChan, doneChan)

	return res
}

func (p *packSeqRes) Creator() anyvec.Creator {
	return p.C
}

func (p *packSeqRes) Forward() <-chan *anyseq.Batch {
	return p.Out
}

func (p *packSeqRes) Vars() anydiff.VarSet {
	<-p.Done
	return p.V
}

func (p *packSeqRes) Propagate(upstream <-chan *anyseq.Batch, grad *Grad) {
	for _ = range p.Forward() {
	}

	downstreams, wg := propagateMany(p.Ins, grad)
	for upBatch := range upstream {
		for i, part := range p.splitUpstream(upBatch) {
			if part != nil && downstreams[i] != nil {
				downstreams[i] <- part
			}
		}
	}

	for _, ch := range downstreams {
		if ch != nil {
			close(ch)
		}
	}

	wg.Wait()
}

func (p *packSeqRes) forward(out chan<- *anyseq.Batch, done chan<- struct{}) {
	c := p.Creator()

	for {
		var numOpen int
		var batches []*anyseq.Batch
		for inIdx, in := range p.Ins {
			batch, ok := <-in.Forward()
			if ok {
				numOpen++
				batches = append(batches, batch)
				p.LanesPerSeq[inIdx] = len(batch.Present)
				p.Lens[inIdx]++
			} else {
				lanes := p.LanesPerSeq[inIdx]
				batches = append(batches, fillerBatch(c, lanes))
			}
		}
		if numOpen == 0 {
			break
		}
		out <- joinBatches(c, batches)
	}

	for _, in := range p.Ins {
		p.V = anydiff.MergeVarSets(p.V, in.Vars())
	}

	close(done)
	close(out)
}

// splitUpstream splits an upstream batch into upstream
// batches for each input.
// If an input is not present yet, its batch is nil.
func (p *packSeqRes) splitUpstream(upBatch *anyseq.Batch) []*anyseq.Batch {
	vecSize := upBatch.Packed.Len() / upBatch.NumPresent()
	res := make([]*anyseq.Batch, len(p.Ins))

	var laneOffset int
	var vecOffset int
	for inIdx, numLanes := range p.LanesPerSeq {
		subBatch := &anyseq.Batch{
			Present: upBatch.Present[laneOffset : laneOffset+numLanes],
		}
		if subBatch.NumPresent() > 0 {
			subBatch.Packed = upBatch.Packed.Slice(vecOffset*vecSize,
				(vecOffset+subBatch.NumPresent())*vecSize)
			res[inIdx] = subBatch
			vecOffset += subBatch.NumPresent()
		}
		laneOffset += numLanes
	}

	return res
}

type packRereaderRes struct {
	*packSeqRes
	Rereaders []Rereader
}

// PackRereader is like Pack, but for Rereaders.
func PackRereader(c anyvec.Creator, rs []Rereader) Rereader {
	plain := make([]Seq, len(rs))
	for i, x := range rs {
		plain[i] = x
	}
	return &packRereaderRes{
		packSeqRes: PackSeq(c, plain).(*packSeqRes),
		Rereaders:  rs,
	}
}

func (p *packRereaderRes) Reread(start, end int) <-chan *anyseq.Batch {
	<-p.packSeqRes.Done

	if start > end || start < 0 {
		panic("slice bounds out of range")
	}

	sourceChans := make([]<-chan *anyseq.Batch, len(p.Ins))
	var maxLen int
	for i, seqLen := range p.Lens {
		maxLen = essentials.MaxInt(maxLen, seqLen)
		if seqLen <= start {
			empty := make(chan *anyseq.Batch)
			sourceChans[i] = empty
			close(empty)
		} else {
			subEnd := end
			if subEnd > seqLen {
				subEnd = seqLen
			}
			sourceChans[i] = p.Rereaders[i].Reread(start, subEnd)
		}
	}

	if end > maxLen {
		panic("slice bounds out of range")
	}

	out := make(chan *anyseq.Batch, 1)

	go func() {
		c := p.Creator()
		streamAndJoin(c, sourceChans, p.LanesPerSeq, out)
		close(out)
	}()

	return out
}

type packedTape struct {
	LanesCounted <-chan struct{}
	Creator      anyvec.Creator
	LanesPerTape []int

	Tapes []Tape
}

// PackTape creates an aggregate Tape that combines the
// sequences from other tapes.
func PackTape(tapes []Tape) Tape {
	countedChan := make(chan struct{})
	res := &packedTape{
		LanesCounted: countedChan,
		LanesPerTape: make([]int, len(tapes)),
		Tapes:        tapes,
	}

	go func() {
		res.countLanes()
		close(countedChan)
	}()

	return res
}

func (t *packedTape) ReadTape(start, end int) <-chan *anyseq.Batch {
	res := make(chan *anyseq.Batch)
	inChans := make([]<-chan *anyseq.Batch, len(t.Tapes))
	for i, t := range t.Tapes {
		inChans[i] = t.ReadTape(start, end)
	}
	go func() {
		<-t.LanesCounted
		// A nil creator means there are no batches, anyway.
		if t.Creator != nil {
			streamAndJoin(t.Creator, inChans, t.LanesPerTape, res)
		}
		close(res)
	}()
	return res
}

func (p *packedTape) countLanes() {
	seqs := make([]<-chan *anyseq.Batch, len(p.Tapes))
	for i, t := range p.Tapes {
		seqs[i] = t.ReadTape(0, 1)
	}
	for i, ch := range seqs {
		if batch, ok := <-ch; ok {
			p.LanesPerTape[i] = len(batch.Present)
			p.Creator = batch.Packed.Creator()
		}
	}
}

// joinBatches concatenates the batches.
// A batch may be from fillerBatch().
func joinBatches(c anyvec.Creator, batches []*anyseq.Batch) *anyseq.Batch {
	var packed []anyvec.Vector
	var present []bool
	for _, batch := range batches {
		// NumPresent is 0 if this is a filler batch.
		if batch.NumPresent() != 0 {
			packed = append(packed, batch.Packed)
		}
		present = append(present, batch.Present...)
	}
	return &anyseq.Batch{
		Packed:  c.Concat(packed...),
		Present: present,
	}
}

// streamAndJoin reads batches from multiple sequences,
// joins them, and sends them to out.
//
// The seqsPerChan argument stores the size of the Present
// list for each source, so that filler batches can be
// created if a source runs out before the rest.
func streamAndJoin(c anyvec.Creator, sources []<-chan *anyseq.Batch,
	seqsPerChan []int, out chan<- *anyseq.Batch) {
	for {
		var batches []*anyseq.Batch
		var gotAny bool
		for inIdx, ch := range sources {
			batch, ok := <-ch
			if ok {
				gotAny = true
				batches = append(batches, batch)
			} else {
				lanes := seqsPerChan[inIdx]
				batches = append(batches, fillerBatch(c, lanes))
			}
		}
		if !gotAny {
			return
		}
		out <- joinBatches(c, batches)
	}
}

// fillerBatch creates a placeholder batch that signifies
// that a sequence batch has ended.
func fillerBatch(c anyvec.Creator, lanes int) *anyseq.Batch {
	return &anyseq.Batch{Present: make([]bool, lanes)}
}

// propagateMany creates downstream channels for each Seq
// and propagates through each of them asynchronously.
// The returned sync.WaitGroup is closed when propagation
// has completed.
//
// If a Seq is constant, its downstream channel will be
// nil and it will not be propagated through.
//
// The caller should close all the non-nil seqs.
func propagateMany(seqs []Seq, grad *Grad) ([]chan<- *anyseq.Batch, *sync.WaitGroup) {
	downstreams := make([]chan<- *anyseq.Batch, len(seqs))
	wg := &sync.WaitGroup{}
	for i, in := range seqs {
		var needProp bool
		grad.Use(func(g anydiff.Grad) {
			needProp = g.Intersects(in.Vars())
		})
		if !needProp {
			continue
		}
		wg.Add(1)
		ch := make(chan *anyseq.Batch, 1)
		go func(in Seq, ch <-chan *anyseq.Batch) {
			in.Propagate(ch, grad)
			wg.Done()
		}(in, ch)
		downstreams[i] = ch
	}
	return downstreams, wg
}
