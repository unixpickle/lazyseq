package lazyseq

import (
	"sync"

	"github.com/unixpickle/anydiff/anyseq"
)

// A Reuser is a Rereader which can be used as a Seq more
// than once by virtue of its Reuse() method.
type Reuser interface {
	Rereader

	// Reuse resets the Forward() channel.
	//
	// While this may be called at any time, it is
	// recommended that you only call it after the
	// previous forward channel has been consumed
	// either explicitly or implicitly (via a
	// Propagate() call).
	//
	// You should not call Reuse() while the old
	// Forward() channel is being used elsewhere.
	Reuse()
}

type reuser struct {
	Rereader

	FwdLock sync.Mutex

	// Fwd is the current forward channel.
	//
	// This may be nil, in which case Reuse() was called
	// but Forward() never was.
	Fwd <-chan *anyseq.Batch

	// SeqLen is a count generated by the first forward
	// channel and then used as an argument to Reread().
	SeqLen int
}

// MakeReuser wraps an unused Rereader in a Reuser.
func MakeReuser(r Rereader) Reuser {
	firstChan := make(chan *anyseq.Batch, 1)
	res := &reuser{
		Rereader: r,
		Fwd:      firstChan,
	}
	go func() {
		for in := range r.Forward() {
			firstChan <- in
			res.SeqLen++
		}
		close(firstChan)
	}()
	return res
}

func (r *reuser) Forward() <-chan *anyseq.Batch {
	r.FwdLock.Lock()
	defer r.FwdLock.Unlock()
	if r.Fwd == nil {
		r.Fwd = r.Rereader.Reread(0, r.SeqLen)
	}
	return r.Fwd
}

func (r *reuser) Propagate(u <-chan *anyseq.Batch, g Grad) {
	r.FwdLock.Lock()
	if r.Fwd != nil {
		for _ = range r.Fwd {
		}
	}
	r.FwdLock.Unlock()

	r.Rereader.Propagate(u, g)
}

func (r *reuser) Reuse() {
	r.FwdLock.Lock()
	if r.Fwd != nil {
		for _ = range r.Fwd {
		}
		r.Fwd = nil
	}
	r.FwdLock.Unlock()
}