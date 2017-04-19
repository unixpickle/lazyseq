package lazyrnn

import (
	"sync"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/lazyseq"
)

// An rnnFragment is the result of running an RNN on a
// sub-range of a sequence.
// It is similar to a Seq, but with more flexible
// back-propagation.
type rnnFragment interface {
	// Forward is like Seq.Forward.
	Forward() <-chan *anyseq.Batch

	// Vars is like Seq.Vars.
	//
	// This cannot report any variables upon which the
	// inputs might depend, since the inputs are given
	// as raw batches.
	Vars() anydiff.VarSet

	// Propagate is similar to Seq.Propagate, but with
	// a few major differences.
	//
	// Unlike in Seq.Propagate, the downstream channel is
	// provided by the caller.
	// Propagate will not close said channel.
	// The downstream channel may be nil.
	//
	// Unlike in Seq.Propagate, the upstream channel may
	// be left open even after Propagate is done with it.
	//
	// The downstream state is returned.
	Propagate(down chan<- *anyseq.Batch, up <-chan *anyseq.Batch,
		stateUp anyrnn.StateGrad, grad lazyseq.Grad) anyrnn.StateGrad
}

// rereaderFragment represents a fragment of a Rereader.
//
// The fragment may be from the forward pass, or the
// forward pass may have already happened.
// The only difference is the source of the Forward field.
type rereaderFragment struct {
	// Offset is the start index in the Rereader.
	Offset int

	// Forward is a stream of inputs.
	// Use once before using the Rereader field.
	Forward <-chan *anyseq.Batch

	Rereader lazyseq.Rereader
}

func rnnFragmentToSeq(in lazyseq.Seq, block anyrnn.Block, r rnnFragment) lazyseq.Seq {
	return &rnnFragSeq{
		In:    in,
		Block: block,
		Frag:  r,
	}
}

type rnnFragSeq struct {
	In    lazyseq.Seq
	Block anyrnn.Block
	Frag  rnnFragment

	VLock sync.Mutex
	V     anydiff.VarSet
}

func (r *rnnFragSeq) Creator() anyvec.Creator {
	return r.In.Creator()
}

func (r *rnnFragSeq) Forward() <-chan *anyseq.Batch {
	return r.Frag.Forward()
}

func (r *rnnFragSeq) Vars() anydiff.VarSet {
	r.VLock.Lock()
	defer r.VLock.Unlock()
	if r.V == nil {
		r.V = anydiff.MergeVarSets(r.Frag.Vars(), r.In.Vars())
	}
	return r.V
}

func (r *rnnFragSeq) Propagate(u <-chan *anyseq.Batch, grad lazyseq.Grad) {
	for _ = range r.Forward() {
	}

	var downstream chan *anyseq.Batch
	grad.Use(func(g anydiff.Grad) {
		if g.Intersects(r.In.Vars()) {
			downstream = make(chan *anyseq.Batch, 1)
		}
	})

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		if downstream != nil {
			defer close(downstream)
		}

		nextGrad := r.Frag.Propagate(downstream, u, nil, grad)
		if nextGrad != nil {
			r.propagateStart(nextGrad, grad)
		}

		if _, ok := <-u; ok {
			panic("too many upstream batches")
		}
	}()

	if downstream != nil {
		r.In.Propagate(downstream, grad)
	}

	wg.Wait()
}

func (r *rnnFragSeq) propagateStart(nextGrad anyrnn.StateGrad, grad lazyseq.Grad) {
	numSeqs := len(nextGrad.Present())
	if nextGrad.Present().NumPresent() != numSeqs {
		allTrue := make(anyrnn.PresentMap, numSeqs)
		for i := range allTrue {
			allTrue[i] = true
		}
		nextGrad = nextGrad.Expand(allTrue)
	}
	grad.Use(func(g anydiff.Grad) {
		r.Block.PropagateStart(nextGrad, g)
	})
}
