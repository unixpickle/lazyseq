package lazyrnn

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anyvec"
)

func bptt(in <-chan *anyseq.Batch, block anyrnn.Block, start anyrnn.State) rnnFragment {
	outChan := make(chan *anyseq.Batch, 1)
	doneChan := make(chan struct{})
	frag := &bpttResult{forward: outChan, done: doneChan}

	go func() {
		state := start
		for batch := range in {
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
		}
		close(outChan)
	}()

	return frag
}

type bpttResult struct {
	forward <-chan *anyseq.Batch

	// Fields are immutable once done is closed.
	done  <-chan struct{}
	reses []anyrnn.Res
}

func (b *bpttResult) Forward() <-chan *anyseq.Batch {
	return b.forward
}

func (b *bpttResult) Propagate(down chan<- *anyseq.Batch, up <-chan *anyseq.Batch,
	stateUp anyrnn.StateGrad, grad *Grad) anyrnn.StateGrad {
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
