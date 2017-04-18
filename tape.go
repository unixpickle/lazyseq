package lazyseq

import (
	"sync"

	"github.com/unixpickle/anydiff/anyseq"
)

// A Tape is a non-differentiable sequence that can be
// accessed randomly and may be generated while it is
// being read.
//
// All of the restrictions on sequences apply to to Tapes.
// For example, all timesteps must have the same number of
// entries in the Present list.
// Also, it is invalid for a sequence to go away (i.e. not
// be present) and then become present again later.
//
// Tapes can be used to store and re-use sequences.
// For example, you can use a Tape to record the results
// of a reinforcement learning episode as it happens.
type Tape interface {
	// ReadTape generates a channel that is sent a range
	// of values from the tape.
	//
	// If end is -1, then the entire Tape is read.
	//
	// If the Tape is still being generated, the resulting
	// channel may be sent new values from the Tape as they
	// are written.
	//
	// If the range is out of bounds even after the Tape
	// is finished being generated, then the out of bounds
	// part of the range is ignored.
	// In other words, the channel will be closed before
	// (end-start) values have been sent.
	ReadTape(start, end int) <-chan *anyseq.Batch
}

// ReferenceTape creates a Tape that stores the outputs by
// retaining references to all of the batches from every
// time-step.
//
// It produces a Tape and a corresponding writer channel.
// The caller can send to the channel to add time-steps to
// the tape, and then close the channel to complete the
// tape.
//
// The caller must close the write channel to free
// resources associated with the Tape.
func ReferenceTape() (Tape, chan<- *anyseq.Batch) {
	return newAbstractTape(func(in interface{}) *anyseq.Batch {
		return in.(*anyseq.Batch)
	}, func(b *anyseq.Batch) interface{} {
		return b
	})
}

// abstractTape is a tape of abstract objects which can be
// converted to and from *anyseq.Batch.
type abstractTape struct {
	lock      sync.Mutex
	timesteps []interface{}
	done      bool
	nextWait  chan struct{}

	toBatch   func(in interface{}) *anyseq.Batch
	fromBatch func(b *anyseq.Batch) interface{}
}

func newAbstractTape(to func(in interface{}) *anyseq.Batch,
	from func(b *anyseq.Batch) interface{}) (Tape, chan<- *anyseq.Batch) {
	res := &abstractTape{
		nextWait:  make(chan struct{}),
		toBatch:   to,
		fromBatch: from,
	}
	inChan := make(chan *anyseq.Batch, 1)
	go res.readInputs(inChan)
	return res, inChan
}

func (a *abstractTape) ReadTape(start, end int) <-chan *anyseq.Batch {
	if start < 0 {
		panic("negative start index")
	} else if end < start && end != -1 {
		panic("invalid end index")
	}

	res := make(chan *anyseq.Batch, 1)
	go func() {
		defer close(res)
		for i := start; i < end || end == -1; i++ {
			a.lock.Lock()
			for i >= len(a.timesteps) {
				if a.done {
					a.lock.Unlock()
					return
				}
				waiter := a.nextWait
				a.lock.Unlock()
				<-waiter
				a.lock.Lock()
			}
			item := a.timesteps[i]
			a.lock.Unlock()
			res <- a.toBatch(item)
		}
	}()
	return res
}

func (a *abstractTape) readInputs(inChan <-chan *anyseq.Batch) {
	var lastPresent []bool
	for input := range inChan {
		if lastPresent == nil {
			lastPresent = input.Present
		} else {
			if len(lastPresent) != len(input.Present) {
				panic("mismatching present map size")
			}
			for i, newPres := range input.Present {
				oldPres := lastPresent[i]
				if !oldPres && newPres {
					panic("absent sequence became present again")
				}
			}
			lastPresent = input.Present
		}
		inObj := a.fromBatch(input)
		a.lock.Lock()
		a.timesteps = append(a.timesteps, inObj)
		close(a.nextWait)
		a.nextWait = make(chan struct{})
		a.lock.Unlock()
	}
	a.lock.Lock()
	a.done = true
	close(a.nextWait)
	a.lock.Unlock()
}
