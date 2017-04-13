package lazyrnn

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anynet/anyrnn"
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
func FixedHSM(intervalSize int, in Rereader, b anyrnn.Block) Seq {
	return RecursiveHSM(intervalSize, intervalSize+1, in, b)
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
func RecursiveHSM(intervalSize, numPartitions int, in Rereader, b anyrnn.Block) Seq {
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
	frag := recHSM(intervalSize, numPartitions, inFrag, b, nil)
	return rnnFragmentToSeq(in, b, frag)
}

// recHSM applies recursive hidden-state memorization
// to a fragment of a sequence.
//
// The start argument may be nil if this is the beginning
// of the sequence.
func recHSM(interval, partitions int, in *rereaderFragment, block anyrnn.Block,
	start anyrnn.State) rnnFragment {
	outChan := make(chan *anyseq.Batch, 1)
	doneChan := make(chan struct{})
	res := &recHSMFrag{
		In:         in,
		Out:        outChan,
		Partitions: partitions,
		Interval:   interval,
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
	stateUp anyrnn.StateGrad, grad *Grad) anyrnn.StateGrad {
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
	// In the case where end-start == r.Partitions, we don't use
	// BPTT because HSM still uses less memory since it doesn't
	// store internal RNN states.
	if end-start < r.Partitions {
		return bptt(inChan, r.Block, state)
	} else {
		// TODO: look into different ways of determining interval,
		// i.e. different rounding strategies.
		interval := (end - start) / r.Partitions
		inFrag := &rereaderFragment{
			Offset:   r.In.Offset + start,
			Forward:  inChan,
			Rereader: r.In.Rereader,
		}
		return recHSM(interval, r.Partitions, inFrag, r.Block, state)
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
