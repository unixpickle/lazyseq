package test

import (
	"reflect"
	"testing"
	"time"

	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/lazyseq"
)

func TestReferenceTape(t *testing.T) {
	tape, writer := lazyseq.ReferenceTape()
	readers := []<-chan *anyseq.Batch{
		tape.ReadTape(1, 3),
		tape.ReadTape(2, 5),
		tape.ReadTape(2, -1),
		tape.ReadTape(0, 2),
	}

	batches := []*anyseq.Batch{
		{Present: []bool{true, false, true}},
		{Present: []bool{true, false, false}},
		{Present: []bool{true, false, false}},
		{Present: []bool{false, false, false}},
	}
	writer <- batches[0]

	mustNotRead(t, readers[:3]...)
	mustRead(t, batches[0], readers[3])

	writer <- batches[1]
	mustRead(t, batches[1], readers[0], readers[3])
	mustNotRead(t, readers[1], readers[2])

	writer <- batches[2]
	mustRead(t, batches[2], readers[:3]...)
	mustRead(t, nil, readers[3])

	writer <- batches[3]
	mustRead(t, batches[3], readers[1:3]...)
	mustRead(t, nil, readers[0], readers[3])

	close(writer)

	mustRead(t, nil, readers...)
}

func mustNotRead(t *testing.T, chans ...<-chan *anyseq.Batch) {
	time.Sleep(time.Millisecond * 50)
	for _, ch := range chans {
		select {
		case <-ch:
			t.Fatal("did not expect to see value")
		default:
		}
	}
}

func mustRead(t *testing.T, expected *anyseq.Batch, chans ...<-chan *anyseq.Batch) {
	for _, ch := range chans {
		timeout := time.After(time.Second)
		select {
		case actual := <-ch:
			if !reflect.DeepEqual(actual, expected) {
				t.Fatalf("expected %v but got %v", expected, actual)
			}
		case <-timeout:
			t.Fatalf("expected to see value: %v", expected)
		}
	}
}
