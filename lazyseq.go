// Package lazyseq provides abstractions and utilities for
// dealing with lazy sequences of differentiable vectors.
// It is designed to be used with recurrent neural
// networks via the lazyrnn sub-package, but that is not
// the only possible use for it.
package lazyseq

import (
	"sync"

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
	// The upstream argument is a channel of upstream
	// gradients, ordered last-to-first timestep.
	// The caller should ensure that u is closed once all
	// upstream batches have been sent.
	// Propagate() may modify upstream vectors, perhaps
	// to use them as scratch space.
	//
	// The Forward() channel should not be used once
	// Propagate() has been called.
	//
	// Propagate should unblock all pending calls to Vars()
	// before it attempts to receive upstream batches.
	//
	// Propagate may be called more than once.
	Propagate(upstream <-chan *anyseq.Batch, grad Grad)
}

// A Rereader is a Seq which can re-produce any sub-range
// of its outputs.
type Rereader interface {
	Seq

	// Reread creates a channel which is sent the outputs
	// in the range [start, end).
	// The channel is closed once all batches are sent.
	//
	// This may block in the same way that Vars() may block.
	// Importantly, the upstream channel passed to
	// Propagate() may depend on Reread(), so Propagate()
	// absolutely must unblock Reread() before attempting to
	// read upstream batches.
	Reread(start, end int) <-chan *anyseq.Batch
}

// Grad is a gradient paired with some sort of
// synchronization method to make it thread-safe.
type Grad interface {
	// Use calls f such that, while f is running, all other
	// calls to Use() block.
	Use(f func(g anydiff.Grad))
}

type mutexGrad struct {
	lock sync.RWMutex
	grad anydiff.Grad
}

// NewGrad creates a Grad which wraps a raw gradient.
func NewGrad(g anydiff.Grad) Grad {
	return &mutexGrad{grad: g}
}

func (m *mutexGrad) Use(f func(g anydiff.Grad)) {
	m.lock.Lock()
	defer m.lock.Unlock()
	f(m.grad)
}
