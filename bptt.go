package lazyrnn

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anyvec"
)

// bptt performs back-propagation through time.
type bptt struct {
	Block anyrnn.Block

	Ins      <-chan *anyseq.Batch
	Upstream <-chan *anyseq.Batch

	// May be nil.
	Downstream chan<- *anyseq.Batch

	Start anyrnn.State
	Grad  anydiff.Grad

	// May be nil.
	UpstreamState anyrnn.StateGrad
}

// Run runs back-propagation through time and returns the
// downstream state gradient.
func (b *bptt) Run() anyrnn.StateGrad {
	state := b.Start

	var reses []anyrnn.Res
	for in := range b.Ins {
		if in.NumPresent() != state.Present().NumPresent() {
			state = state.Reduce(in.Present)
		}
		res := b.Block.Step(state, in.Packed)
		reses = append(reses, res)
		state = res.State()
	}

	nextGrad := b.UpstreamState
	for j := len(reses) - 1; j >= 0; j-- {
		res := reses[j]
		pres := res.State().Present()
		if nextGrad != nil && nextGrad.Present().NumPresent() != pres.NumPresent() {
			nextGrad = nextGrad.Expand(pres)
		}
		var inDown anyvec.Vector
		upBatch, ok := <-b.Upstream
		if !ok {
			panic("not enough upstream batches")
		}
		upVec := upBatch.Packed
		inDown, nextGrad = res.Propagate(upVec, nextGrad, b.Grad)
		if b.Downstream != nil {
			b.Downstream <- &anyseq.Batch{
				Packed:  inDown,
				Present: upBatch.Present,
			}
		}
	}

	return nextGrad
}
