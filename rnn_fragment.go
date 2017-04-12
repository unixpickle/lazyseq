package lazyrnn

import (
	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anynet/anyrnn"
)

// An rnnFragment is the result of running an RNN on a
// sub-range of a sequence.
// It is similar to a Seq, but with more flexible
// back-propagation.
type rnnFragment interface {
	// Forward is like Seq.Forward.
	Forward() <-chan *anyseq.Batch

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
		stateUp anyrnn.StateGrad, grad *Grad) anyrnn.StateGrad
}
