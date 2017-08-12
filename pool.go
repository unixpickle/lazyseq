package lazyseq

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anyvec"
)

type poolToVecRes struct {
	In       Seq
	Res      anydiff.Res
	UsedVars anydiff.VarSet

	PoolVars []*anydiff.Var
	Presents [][]bool
}

// PoolToVec wraps s and passes the wrapper to f in such a
// way that the result of f will only propagate once
// through s.
//
// The result retains a reference to every timestep of s.
// Thus, there is an implicit size constraint on s.
//
// This is similar to anyseq.PoolToVec.
func PoolToVec(s Seq, f func(anyseq.Seq) anydiff.Res) anydiff.Res {
	res := &poolToVecRes{In: s}

	var resBatches []*anyseq.ResBatch
	for timestep := range s.Forward() {
		v := anydiff.NewVar(timestep.Packed)
		res.PoolVars = append(res.PoolVars, v)
		res.Presents = append(res.Presents, timestep.Present)
		resBatches = append(resBatches, &anyseq.ResBatch{
			Packed:  v,
			Present: timestep.Present,
		})
	}
	pooledSeq := anyseq.ResSeq(s.Creator(), resBatches)
	res.Res = f(pooledSeq)

	// Keep our set of variables correct when f ignores
	// its input entirely.
	indepOfInput := true
	for _, v := range res.PoolVars {
		if res.Res.Vars().Has(v) {
			indepOfInput = false
			break
		}
	}
	if indepOfInput {
		return res.Res
	}

	res.UsedVars = anydiff.MergeVarSets(s.Vars(), res.Res.Vars())
	for _, v := range res.PoolVars {
		res.UsedVars.Del(v)
	}

	return res
}

func (p *poolToVecRes) Vars() anydiff.VarSet {
	return p.UsedVars
}

func (p *poolToVecRes) Output() anyvec.Vector {
	return p.Res.Output()
}

func (p *poolToVecRes) Propagate(u anyvec.Vector, g anydiff.Grad) {
	for _, v := range p.PoolVars {
		g[v] = v.Vector.Creator().MakeVector(v.Vector.Len())
	}

	p.Res.Propagate(u, g)

	upstream := make(chan *anyseq.Batch, len(p.PoolVars))
	for i := len(p.PoolVars) - 1; i >= 0; i-- {
		v := p.PoolVars[i]
		upstream <- &anyseq.Batch{
			Packed:  g[v],
			Present: p.Presents[i],
		}
		delete(g, v)
	}
	close(upstream)

	p.In.Propagate(upstream, NewGrad(g))
}
