package lazyrnn

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anyvec"
)

type lazifySeq struct {
	Seq anyseq.Seq
	Out <-chan *anyseq.Batch
}

// Lazify creates a lazy sequence out of an anyseq.Seq.
func Lazify(seq anyseq.Seq) Rereader {
	out := seq.Output()
	outChan := make(chan *anyseq.Batch, len(out))
	for _, x := range seq.Output() {
		outChan <- x
	}
	close(outChan)
	return &lazifySeq{Seq: seq, Out: outChan}
}

func (l *lazifySeq) Creator() anyvec.Creator {
	return l.Seq.Creator()
}

func (l *lazifySeq) Forward() <-chan *anyseq.Batch {
	return l.Out
}

func (l *lazifySeq) Vars() anydiff.VarSet {
	return l.Seq.Vars()
}

func (l *lazifySeq) Propagate(upstream <-chan *anyseq.Batch, grad anydiff.Grad) {
	uList := make([]*anyseq.Batch, len(l.Seq.Output()))
	for i := len(uList) - 1; i >= 0; i-- {
		var ok bool
		uList[i], ok = <-upstream
		if !ok {
			panic("not enough upstream batches")
		}
	}
	if _, ok := <-upstream; ok {
		panic("too many upstream batches")
	}
	l.Seq.Propagate(uList, grad)
}

func (l *lazifySeq) Reread(start, end int) <-chan *anyseq.Batch {
	res := make(chan *anyseq.Batch, end-start)
	for _, x := range l.Seq.Output()[start:end] {
		res <- x
	}
	close(res)
	return res
}

type unlazifySeq struct {
	Seq  Seq
	Outs []*anyseq.Batch
	V    anydiff.VarSet
}

// Unlazify creates an anyseq.Seq by fully reading a Seq.
//
// The seq argument should be an unread Seq.
// This will not work if seq.Forward() has been read from
// or if seq.Propagate() has been called.
func Unlazify(seq Seq) anyseq.Seq {
	var outs []*anyseq.Batch
	for out := range seq.Forward() {
		outs = append(outs, out)
	}
	return &unlazifySeq{Seq: seq, Outs: outs, V: seq.Vars()}
}

func (u *unlazifySeq) Creator() anyvec.Creator {
	return u.Seq.Creator()
}

func (u *unlazifySeq) Output() []*anyseq.Batch {
	return u.Outs
}

func (u *unlazifySeq) Vars() anydiff.VarSet {
	return u.V
}

func (u *unlazifySeq) Propagate(upstream []*anyseq.Batch, grad anydiff.Grad) {
	uChan := make(chan *anyseq.Batch, len(upstream))
	for i := len(upstream) - 1; i >= 0; i-- {
		uChan <- upstream[i]
	}
	close(uChan)
	u.Seq.Propagate(uChan, grad)
}
