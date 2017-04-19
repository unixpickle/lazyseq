package lazyrnn

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anynet/anyrnn"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/lazyseq"
)

// FixedHSM uses fixed-interval hidden state memorization
// to apply the RNN block to the sequence.
// Every intervalSize timesteps, a hidden state is saved.
//
// If the sequence length T is known beforehand, the best
// intervalSize is sqrt(T).
// In this case, the algorithm is equivalent to Chen's
// sqrt(T) algorithm, and it will use O(sqrt(T)) memory.
// See https://arxiv.org/abs/1604.06174.
//
// If lazyBPTT is true, then back-propagation will never
// store more internal states or inputs than it needs to.
func FixedHSM(intervalSize int, lazyBPTT bool, in lazyseq.Rereader,
	b anyrnn.Block) lazyseq.Seq {
	return RecursiveHSM(intervalSize, intervalSize+1, lazyBPTT, in, b)
}

// RecursiveHSM uses recursive hidden state memorization
// to apply the RNN block to the sequence.
// Every intervalSize timesteps, a hidden state is saved.
// When back-propagating through each sub-sequence of
// length intervalSize, the sub-sequence will be divided
// into numPartitions pieces and back-propagation will
// proceed recursively.
//
// If the sequence length T is known beforehand, it is
// recommended to use T/numPartitions as the intervalSize.
// In this case, the algorithm uses O(log(T)) memory and
// O(T*log(T)) time.
//
// If lazyBPTT is true, then back-propagation will never
// store more internal states or inputs than it needs to.
func RecursiveHSM(intervalSize, numPartitions int, lazyBPTT bool,
	in lazyseq.Rereader, b anyrnn.Block) lazyseq.Seq {
	if intervalSize < 1 {
		panic("invalid interval size")
	}
	if numPartitions < 2 {
		panic("invalid number of partitions")
	}
	inFrag := &rereaderFragment{
		Forward:  in.Forward(),
		Rereader: in,
	}
	frag := recHSM(intervalSize, numPartitions, lazyBPTT, inFrag, b, nil)
	return rnnFragmentToSeq(in, b, frag)
}

// recHSM applies recursive hidden-state memorization
// to a fragment of a sequence.
//
// The start argument may be nil if this is the beginning
// of the sequence.
func recHSM(interval, partitions int, lazyBPTT bool, in *rereaderFragment,
	block anyrnn.Block, start anyrnn.State) rnnFragment {
	outChan := make(chan *anyseq.Batch, 1)
	doneChan := make(chan struct{})
	res := &recHSMFrag{
		In:         in,
		Out:        outChan,
		Partitions: partitions,
		Interval:   interval,
		LazyBPTT:   lazyBPTT,
		Block:      block,
		Done:       doneChan,
	}
	go res.forward(outChan, doneChan, start)
	return res
}

// recHSMFrag is an rnnFragment for recursive HSM.
type recHSMFrag struct {
	In         *rereaderFragment
	Out        <-chan *anyseq.Batch
	Partitions int
	Interval   int
	LazyBPTT   bool
	Block      anyrnn.Block

	// Fields become valid after done is closed.
	Done     <-chan struct{}
	Saved    []anyrnn.State
	V        anydiff.VarSet
	NumSteps int
}

func (r *recHSMFrag) Forward() <-chan *anyseq.Batch {
	return r.Out
}

func (r *recHSMFrag) Vars() anydiff.VarSet {
	<-r.Done
	return r.V
}

func (r *recHSMFrag) Propagate(down chan<- *anyseq.Batch, up <-chan *anyseq.Batch,
	stateUp anyrnn.StateGrad, grad lazyseq.Grad) anyrnn.StateGrad {
	for _ = range r.Forward() {
	}

	nextGrad := stateUp
	nextIdx := r.NumSteps
	for i := len(r.Saved) - 1; i >= 0; i-- {
		frag := r.subFragment(i*r.Interval, nextIdx, r.Saved[i])
		nextGrad = frag.Propagate(down, up, nextGrad, grad)
		nextIdx = i * r.Interval
	}
	return nextGrad
}

func (r *recHSMFrag) subFragment(start, end int, state anyrnn.State) rnnFragment {
	inChan := r.In.Rereader.Reread(start+r.In.Offset, end+r.In.Offset)
	if end-start <= 1 || (end-start <= r.Partitions && !r.LazyBPTT) {
		return bptt(inChan, r.Block, state)
	} else {
		// TODO: look into different ways of determining interval,
		// i.e. different rounding strategies.
		interval := essentials.MaxInt(1, (end-start)/r.Partitions)
		inFrag := &rereaderFragment{
			Offset:   r.In.Offset + start,
			Forward:  inChan,
			Rereader: r.In.Rereader,
		}
		return recHSM(interval, r.Partitions, r.LazyBPTT, inFrag, r.Block, state)
	}
}

func (r *recHSMFrag) forward(outChan chan<- *anyseq.Batch, doneChan chan<- struct{},
	state anyrnn.State) {
	for input := range r.In.Forward {
		if state == nil {
			state = r.Block.Start(len(input.Present))
		}
		if state.Present().NumPresent() != input.NumPresent() {
			state = state.Reduce(input.Present)
		}
		if r.NumSteps%r.Interval == 0 {
			r.Saved = append(r.Saved, state)
		}
		r.NumSteps++

		res := r.Block.Step(state, input.Packed)
		r.V = anydiff.MergeVarSets(r.V, res.Vars())
		state = res.State()
		outChan <- &anyseq.Batch{Present: input.Present, Packed: res.Output()}
	}

	close(doneChan)
	close(outChan)
}
