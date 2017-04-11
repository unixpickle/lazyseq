// Package lazyrnn evaluates and trains RNNs on limited
// memory systems.
package lazyrnn

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anyvec"
)

// Seq is a lazily-evaluated sequence.
type Seq interface {
	// Creator returns the anyvec.Creator associated with
	// the sequence.
	Creator() anyvec.Creator

	// Forward returns a channel of sequence outputs,
	// starting with the first output.
	// The channel is closed once all outputs have been
	// sent.
	//
	// The same channel is returned every time.
	// The channel may not be used during or after
	// back-propagation.
	Forward() <-chan *anyseq.Batch

	// Vars returns the variables upon which the sequence
	// depends.
	//
	// Since the variables may not be known until the
	// entire sequence has been evaluated, this may block
	// until the Forward() sequence has been fully read or
	// until back-propagation has been performed.
	Vars() anydiff.VarSet

	// Propagate performs back-propagation.
	//
	// The u argument is a channel of upstream gradients,
	// ordered from the last timestep to the first.
	// The caller should ensure that u is closed once all
	// upstream batches have been sent.
	//
	// The Forward() channel should not be used once
	// Propagate has been called.
	//
	// A call to Propagate should unblock all pending calls
	// to Vars().
	Propagate(upstream <-chan *anyseq.Batch, grad anydiff.Grad)
}
