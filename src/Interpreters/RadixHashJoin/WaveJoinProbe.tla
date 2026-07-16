---------------------------- MODULE WaveJoinProbe ----------------------------
EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS
    Input, HashValue, Result, Error,
    Bytes(_), Pass0(_), LeafOf(_), ProbeResult(_, _),
    P0, PL, BUDGET, WORKERS,
    FailLeaves, ErrorOf(_),
    NoRow, NoResult, NoError, PreError,
    DisjointRowWritesSafe, LeafHTIsolated,
    ProbeResultRetainable, FreeNoThrow

VARIABLE st

vars == <<st>>

Passes == 0 .. (P0 - 1)
Leaves == 0 .. (P0 * PL - 1)
BlockIds == [i \in 1 .. Len(Input) |-> i]

SeqSet(s) == {s[i] : i \in DOMAIN s}
EmptySeq(n) == [i \in 1 .. n |-> NoRow]
Min(a, b) == IF a <= b THEN a ELSE b

BlockOccs(b) == [r \in 1 .. Len(Input[b]) |-> <<b, r>>]

RECURSIVE RowsOfBlocks(_)
RowsOfBlocks(bs) ==
    IF Len(bs) = 0
    THEN <<>>
    ELSE BlockOccs(Head(bs)) \o RowsOfBlocks(Tail(bs))

Occ == SeqSet(RowsOfBlocks(BlockIds))
HashOf(o) == Input[o[1]][o[2]]
Pid(o) == Pass0(HashOf(o))
Lid(o) == Pid(o) * PL + LeafOf(HashOf(o))

RECURSIVE SumBytes(_)
SumBytes(bs) ==
    IF Len(bs) = 0
    THEN 0
    ELSE Bytes(Head(bs)) + SumBytes(Tail(bs))

Cut(bs) ==
    LET candidates == {k \in 1 .. Len(bs) :
                          SumBytes(SubSeq(bs, 1, k)) >= BUDGET}
    IN  IF candidates = {}
        THEN Len(bs)
        ELSE CHOOSE k \in candidates :
                 \A j \in candidates : k <= j

RECURSIVE Waves(_)
Waves(bs) ==
    IF Len(bs) = 0
    THEN <<>>
    ELSE LET k == Cut(bs)
         IN  <<SubSeq(bs, 1, k)>>
             \o Waves(SubSeq(bs, k + 1, Len(bs)))

CanonicalWaves == Waves(BlockIds)

RECURSIVE PassPart(_, _)
PassPart(rs, p) ==
    IF Len(rs) = 0
    THEN <<>>
    ELSE IF Pid(Head(rs)) = p
         THEN <<Head(rs)>> \o PassPart(Tail(rs), p)
         ELSE PassPart(Tail(rs), p)

RECURSIVE LeafPart(_, _)
LeafPart(rs, l) ==
    IF Len(rs) = 0
    THEN <<>>
    ELSE IF Lid(Head(rs)) = l
         THEN <<Head(rs)>> \o LeafPart(Tail(rs), l)
         ELSE LeafPart(Tail(rs), l)

ExpectedPass(bs, p) == PassPart(RowsOfBlocks(bs), p)
ExpectedLeaf(bs, l) == LeafPart(RowsOfBlocks(bs), l)

Pos(rs, o) == CHOOSE i \in DOMAIN rs : rs[i] = o

Rank0(rs, o) ==
    Cardinality({i \in 1 .. Pos(rs, o) : Pid(rs[i]) = Pid(o)})

Rank1(rs, o) ==
    Cardinality({i \in 1 .. Pos(rs, o) : Lid(rs[i]) = Lid(o)})

RECURSIVE LeavesOutput(_, _)
LeavesOutput(bs, n) ==
    IF n = 0
    THEN <<>>
    ELSE LeavesOutput(bs, n - 1)
         \o ProbeResult(n - 1, ExpectedLeaf(bs, n - 1))

RECURSIVE WavesOutput(_)
WavesOutput(ws) ==
    IF Len(ws) = 0
    THEN <<>>
    ELSE WavesOutput(SubSeq(ws, 1, Len(ws) - 1))
         \o LeavesOutput(ws[Len(ws)], P0 * PL)

RECURSIVE TotalBlocks(_)
TotalBlocks(ws) ==
    IF Len(ws) = 0
    THEN 0
    ELSE Len(Head(ws)) + TotalBlocks(Tail(ws))

FirstN(s, n) == [i \in 1 .. n |-> s[i]]
IsPrefix(a, b) ==
    /\ Len(a) <= Len(b)
    /\ \A i \in DOMAIN a : a[i] = b[i]

CurrentCanonicalWave ==
    IF Len(st.doneWaves) < Len(CanonicalWaves)
    THEN CanonicalWaves[Len(st.doneWaves) + 1]
    ELSE <<>>

WaveRows == RowsOfBlocks(st.queue)
Outstanding ==
    {l \in Leaves : st.probeState[l] \in {"run", "ok", "err"}}

PreJob(p) == [kind |-> "pre", id |-> p]
ScatterJob(o) == [kind |-> "scatter", id |-> o]
RefAllocJob(p) == [kind |-> "refalloc", id |-> p]
RefineJob(o) == [kind |-> "refine", id |-> o]
ProbeJob(l) == [kind |-> "probe", id |-> l]

AllJobs ==
    {PreJob(p) : p \in Passes}
    \cup {ScatterJob(o) : o \in Occ}
    \cup {RefAllocJob(p) : p \in Passes}
    \cup {RefineJob(o) : o \in Occ}
    \cup {ProbeJob(l) : l \in Leaves}

LeavesOfPass(p) == {p * PL + q : q \in 0 .. (PL - 1)}

Footprint(j) ==
    CASE j.kind = "pre" -> {<<"a0", j.id>>}
    []   j.kind = "scatter" ->
             {<<"entry-row", j.id>>,
              <<"a0-cell", Pid(j.id), Rank0(WaveRows, j.id)>>}
    []   j.kind = "refalloc" ->
             {<<"a1", l>> : l \in LeavesOfPass(j.id)}
    []   j.kind = "refine" ->
             {<<"a0-cell", Pid(j.id), Rank0(WaveRows, j.id)>>,
              <<"a1-cell", Lid(j.id), Rank1(WaveRows, j.id)>>}
    []   j.kind = "probe" ->
             {<<"a1", j.id>>, <<"ht", j.id>>, <<"result", j.id>>}
    []   OTHER -> {}

DisjointFromRunning(j) ==
    \A k \in st.running : Footprint(j) \cap Footprint(k) = {}

EligibleNonProbe ==
    CASE st.phase = "pre" ->
             {PreJob(p) : p \in Passes \ st.preDone} \ st.running
    []   st.phase = "scatter" ->
             {ScatterJob(o) : o \in SeqSet(WaveRows) \ st.scatterDone}
             \ st.running
    []   st.phase = "refalloc" ->
             {RefAllocJob(p) : p \in Passes \ st.refAllocDone}
             \ st.running
    []   st.phase = "refine" ->
             {RefineJob(o) : o \in SeqSet(WaveRows) \ st.refineDone}
             \ st.running
    []   OTHER -> {}

Init ==
    st = [phase |-> "acc",
          nextBlock |-> 1,
          queue |-> <<>>,
          mem |-> 0,
          doneWaves |-> <<>>,
          hashCount |-> [o \in Occ |-> 0],
          arena0 |-> [p \in Passes |-> <<>>],
          arena1 |-> [l \in Leaves |-> <<>>],
          preDone |-> {},
          scatterDone |-> {},
          refAllocDone |-> {},
          refineDone |-> {},
          running |-> {},
          probeState |-> [l \in Leaves |-> "idle"],
          probeResult |-> [l \in Leaves |-> NoResult],
          probeError |-> [l \in Leaves |-> NoError],
          nextAdmit |-> 0,
          nextEmit |-> 0,
          admitLimit |-> P0 * PL,
          output |-> <<>>,
          waveOutputPrefix |-> 0,
          primary |-> NoError,
          liveEntries |-> {},
          liveA0 |-> {},
          liveA1 |-> {},
          liveResults |-> {}]

Accumulate ==
    /\ st.phase = "acc"
    /\ st.nextBlock <= Len(Input)
    /\ LET b == st.nextBlock
           close == st.mem + Bytes(b) >= BUDGET
       IN  st' = [st EXCEPT
              !.nextBlock = @ + 1,
              !.queue = Append(@, b),
              !.mem = @ + Bytes(b),
              !.hashCount =
                  [o \in Occ |->
                      IF o \in SeqSet(BlockOccs(b))
                      THEN st.hashCount[o] + 1
                      ELSE st.hashCount[o]],
              !.liveEntries = @ \cup {b},
              !.phase = IF close THEN "pre" ELSE "acc",
              !.waveOutputPrefix =
                  IF close THEN Len(st.output) ELSE @]

CloseFinal ==
    /\ st.phase = "acc"
    /\ st.nextBlock = Len(Input) + 1
    /\ Len(st.queue) > 0
    /\ st' = [st EXCEPT
           !.phase = "pre",
           !.waveOutputPrefix = Len(st.output)]

FinishInput ==
    /\ st.phase = "acc"
    /\ st.nextBlock = Len(Input) + 1
    /\ st.queue = <<>>
    /\ st' = [st EXCEPT !.phase = "done"]

AccFault ==
    /\ st.phase = "acc"
    /\ st.nextBlock <= Len(Input)
    /\ st' = [st EXCEPT
           !.phase = "failed",
           !.primary = PreError,
           !.liveEntries = {},
           !.liveA0 = {},
           !.liveA1 = {},
           !.liveResults = {}]

StartJob(j) ==
    /\ j \in EligibleNonProbe
    /\ Cardinality(st.running) < WORKERS
    /\ DisjointFromRunning(j)
    /\ st' = [st EXCEPT !.running = @ \cup {j}]

FinishPre(p) ==
    LET j == PreJob(p)
    IN  /\ j \in st.running
        /\ st' = [st EXCEPT
             !.running = @ \ {j},
             !.preDone = @ \cup {p},
             !.arena0[p] = EmptySeq(Len(ExpectedPass(st.queue, p))),
             !.liveA0 = @ \cup {p}]

PreBarrier ==
    /\ st.phase = "pre"
    /\ st.preDone = Passes
    /\ st.running = {}
    /\ st' = [st EXCEPT !.phase = "scatter"]

FinishScatter(o) ==
    LET j == ScatterJob(o)
        p == Pid(o)
        k == Rank0(WaveRows, o)
    IN  /\ j \in st.running
        /\ st.arena0[p][k] = NoRow
        /\ st' = [st EXCEPT
             !.running = @ \ {j},
             !.scatterDone = @ \cup {o},
             !.arena0[p][k] = o]

ScatterBarrier ==
    /\ st.phase = "scatter"
    /\ st.scatterDone = SeqSet(WaveRows)
    /\ st.running = {}
    /\ IF PL = 1
       THEN st' = [st EXCEPT
                !.phase = "probe",
                !.arena1 = [l \in Leaves |-> st.arena0[l]],
                !.liveEntries = {},
                !.liveA0 = {},
                !.liveA1 = Leaves]
       ELSE st' = [st EXCEPT
                !.phase = "refalloc",
                !.liveEntries = {}]

FinishRefAlloc(p) ==
    LET j == RefAllocJob(p)
        ls == LeavesOfPass(p)
    IN  /\ j \in st.running
        /\ st' = [st EXCEPT
             !.running = @ \ {j},
             !.refAllocDone = @ \cup {p},
             !.arena1 =
                 [l \in Leaves |->
                     IF l \in ls
                     THEN EmptySeq(Len(ExpectedLeaf(st.queue, l)))
                     ELSE st.arena1[l]],
             !.liveA1 = @ \cup ls]

RefAllocBarrier ==
    /\ st.phase = "refalloc"
    /\ st.refAllocDone = Passes
    /\ st.running = {}
    /\ st' = [st EXCEPT !.phase = "refine"]

FinishRefine(o) ==
    LET j == RefineJob(o)
        l == Lid(o)
        k == Rank1(WaveRows, o)
    IN  /\ j \in st.running
        /\ st.arena1[l][k] = NoRow
        /\ st' = [st EXCEPT
             !.running = @ \ {j},
             !.refineDone = @ \cup {o},
             !.arena1[l][k] = o]

RefineBarrier ==
    /\ st.phase = "refine"
    /\ st.refineDone = SeqSet(WaveRows)
    /\ st.running = {}
    /\ st' = [st EXCEPT
           !.phase = "probe",
           !.liveA0 = {}]

NonProbeFault(j) ==
    /\ st.phase \in {"pre", "scatter", "refalloc", "refine"}
    /\ j \in st.running
    /\ st' = [st EXCEPT
           !.phase = "cancelPre",
           !.running = @ \ {j},
           !.primary = PreError]

CancelNonProbe(j) ==
    /\ st.phase = "cancelPre"
    /\ j \in st.running
    /\ st' = [st EXCEPT !.running = @ \ {j}]

FinishNonProbeFailure ==
    /\ st.phase = "cancelPre"
    /\ st.running = {}
    /\ st' = [st EXCEPT
           !.phase = "failed",
           !.liveEntries = {},
           !.liveA0 = {},
           !.liveA1 = {},
           !.liveResults = {}]

StartProbe(l) ==
    LET j == ProbeJob(l)
    IN  /\ st.phase = "probe"
        /\ l = st.nextAdmit
        /\ l \in Leaves
        /\ l < st.admitLimit
        /\ st.probeState[l] = "idle"
        /\ l \in st.liveA1
        /\ Cardinality(st.running) < WORKERS
        /\ Cardinality(Outstanding) < WORKERS
        /\ DisjointFromRunning(j)
        /\ st' = [st EXCEPT
             !.running = @ \cup {j},
             !.probeState[l] = "run",
             !.nextAdmit = @ + 1]

FinishProbe(l) ==
    LET j == ProbeJob(l)
    IN  /\ j \in st.running
        /\ IF l \in FailLeaves
           THEN st' = [st EXCEPT
                    !.running = @ \ {j},
                    !.probeState[l] = "err",
                    !.probeError[l] = ErrorOf(l),
                    !.admitLimit = Min(@, l + 1)]
           ELSE st' = [st EXCEPT
                    !.running = @ \ {j},
                    !.probeState[l] = "ok",
                    !.probeResult[l] =
                        ProbeResult(l, ExpectedLeaf(st.queue, l)),
                    !.liveResults = @ \cup {l}]

EmitLeaf(l) ==
    /\ st.phase = "probe"
    /\ l = st.nextEmit
    /\ st.probeState[l] = "ok"
    /\ st' = [st EXCEPT
           !.output = @ \o st.probeResult[l],
           !.probeState[l] = "emitted",
           !.probeResult[l] = NoResult,
           !.nextEmit = @ + 1,
           !.liveResults = @ \ {l},
           !.liveA1 = @ \ {l}]

ObserveProbeFailure(l) ==
    /\ st.phase = "probe"
    /\ l = st.nextEmit
    /\ st.probeState[l] = "err"
    /\ st' = [st EXCEPT
           !.phase = "cancelProbe",
           !.primary = st.probeError[l],
           !.probeState =
               [x \in Leaves |->
                   IF x > l /\ st.probeState[x] # "run"
                   THEN "cancel"
                   ELSE st.probeState[x]],
           !.liveResults = {x \in st.liveResults : x <= l}]

CancelProbe(l) ==
    LET j == ProbeJob(l)
    IN  /\ st.phase = "cancelProbe"
        /\ j \in st.running
        /\ st' = [st EXCEPT
             !.running = @ \ {j},
             !.probeState[l] = "cancel"]

FinishProbeFailure ==
    /\ st.phase = "cancelProbe"
    /\ st.running = {}
    /\ st' = [st EXCEPT
           !.phase = "failed",
           !.liveEntries = {},
           !.liveA0 = {},
           !.liveA1 = {},
           !.liveResults = {}]

CompleteWave ==
    /\ st.phase = "probe"
    /\ st.nextEmit = P0 * PL
    /\ st.running = {}
    /\ st' = [st EXCEPT
           !.phase = "acc",
           !.queue = <<>>,
           !.mem = 0,
           !.doneWaves = Append(@, st.queue),
           !.arena0 = [p \in Passes |-> <<>>],
           !.arena1 = [l \in Leaves |-> <<>>],
           !.preDone = {},
           !.scatterDone = {},
           !.refAllocDone = {},
           !.refineDone = {},
           !.probeState = [l \in Leaves |-> "idle"],
           !.probeResult = [l \in Leaves |-> NoResult],
           !.probeError = [l \in Leaves |-> NoError],
           !.nextAdmit = 0,
           !.nextEmit = 0,
           !.admitLimit = P0 * PL,
           !.waveOutputPrefix = Len(st.output),
           !.primary = NoError,
           !.liveEntries = {},
           !.liveA0 = {},
           !.liveA1 = {},
           !.liveResults = {}]

Next ==
    \/ Accumulate
    \/ CloseFinal
    \/ FinishInput
    \/ AccFault
    \/ \E j \in AllJobs : StartJob(j)
    \/ \E p \in Passes : FinishPre(p)
    \/ PreBarrier
    \/ \E o \in Occ : FinishScatter(o)
    \/ ScatterBarrier
    \/ \E p \in Passes : FinishRefAlloc(p)
    \/ RefAllocBarrier
    \/ \E o \in Occ : FinishRefine(o)
    \/ RefineBarrier
    \/ \E j \in AllJobs : NonProbeFault(j)
    \/ \E j \in AllJobs : CancelNonProbe(j)
    \/ FinishNonProbeFailure
    \/ \E l \in Leaves : StartProbe(l)
    \/ \E l \in Leaves : FinishProbe(l)
    \/ \E l \in Leaves : EmitLeaf(l)
    \/ \E l \in Leaves : ObserveProbeFailure(l)
    \/ \E l \in Leaves : CancelProbe(l)
    \/ FinishProbeFailure
    \/ CompleteWave

Spec == Init /\ [][Next]_vars

EnvironmentOK ==
    /\ P0 \in Nat \ {0}
    /\ PL \in Nat \ {0}
    /\ BUDGET \in Nat
    /\ WORKERS \in Nat \ {0}
    /\ HashValue # {}
    /\ Result # {}
    /\ Error # {}
    /\ Input \in Seq(Seq(HashValue))
    /\ \A b \in 1 .. Len(Input) : Bytes(b) \in Nat
    /\ \A h \in HashValue : Pass0(h) \in Passes
    /\ \A h \in HashValue : LeafOf(h) \in 0 .. (PL - 1)
    /\ \A bs \in SeqSet(CanonicalWaves) :
           \A l \in Leaves :
               ProbeResult(l, ExpectedLeaf(bs, l)) \in Seq(Result)
    /\ FailLeaves \subseteq Leaves
    /\ \A l \in Leaves : ErrorOf(l) \in Error
    /\ PreError \in Error
    /\ NoError \notin Error
    /\ NoRow \notin Occ
    /\ NoResult \notin Seq(Result)
    /\ DisjointRowWritesSafe
    /\ LeafHTIsolated
    /\ ProbeResultRetainable
    /\ FreeNoThrow

ASSUME EnvironmentOK

Phases ==
    {"acc", "pre", "scatter", "refalloc", "refine", "probe",
     "cancelPre", "cancelProbe", "done", "failed"}

ProbeStates == {"idle", "run", "ok", "err", "emitted", "cancel"}

TypeOK ==
    /\ st.phase \in Phases
    /\ st.nextBlock \in 1 .. (Len(Input) + 1)
    /\ st.queue \in Seq(1 .. Len(Input))
    /\ st.mem \in Nat
    /\ st.doneWaves \in Seq(Seq(1 .. Len(Input)))
    /\ st.hashCount \in [Occ -> 0 .. 1]
    /\ st.arena0 \in [Passes -> Seq(Occ \cup {NoRow})]
    /\ st.arena1 \in [Leaves -> Seq(Occ \cup {NoRow})]
    /\ st.preDone \subseteq Passes
    /\ st.scatterDone \subseteq Occ
    /\ st.refAllocDone \subseteq Passes
    /\ st.refineDone \subseteq Occ
    /\ st.running \subseteq AllJobs
    /\ st.probeState \in [Leaves -> ProbeStates]
    /\ st.probeResult \in [Leaves -> (Seq(Result) \cup {NoResult})]
    /\ st.probeError \in [Leaves -> (Error \cup {NoError})]
    /\ st.nextAdmit \in 0 .. (P0 * PL)
    /\ st.nextEmit \in 0 .. (P0 * PL)
    /\ st.admitLimit \in 0 .. (P0 * PL)
    /\ st.output \in Seq(Result)
    /\ st.waveOutputPrefix \in Nat
    /\ st.primary \in Error \cup {NoError}
    /\ st.liveEntries \subseteq 1 .. Len(Input)
    /\ st.liveA0 \subseteq Passes
    /\ st.liveA1 \subseteq Leaves
    /\ st.liveResults \subseteq Leaves

WaveExact ==
    /\ Len(st.doneWaves) <= Len(CanonicalWaves)
    /\ st.doneWaves = FirstN(CanonicalWaves, Len(st.doneWaves))
    /\ IsPrefix(st.queue, CurrentCanonicalWave)
    /\ st.nextBlock = TotalBlocks(st.doneWaves) + Len(st.queue) + 1
    /\ st.mem = SumBytes(st.queue)
    /\ st.phase \in
           {"pre", "scatter", "refalloc", "refine", "probe",
            "cancelPre", "cancelProbe"}
       => st.queue = CurrentCanonicalWave
    /\ st.phase = "done" => st.doneWaves = CanonicalWaves

HashExactlyOnce ==
    \A o \in Occ :
        st.hashCount[o] = IF o[1] < st.nextBlock THEN 1 ELSE 0

RankInjective ==
    /\ \A o1, o2 \in SeqSet(WaveRows) :
           /\ Pid(o1) = Pid(o2)
           /\ Rank0(WaveRows, o1) = Rank0(WaveRows, o2)
           => o1 = o2
    /\ \A o1, o2 \in SeqSet(WaveRows) :
           /\ Lid(o1) = Lid(o2)
           /\ Rank1(WaveRows, o1) = Rank1(WaveRows, o2)
           => o1 = o2

CellSafety ==
    /\ \A p \in Passes :
           \A i \in DOMAIN st.arena0[p] :
               \/ st.arena0[p][i] = NoRow
               \/ st.arena0[p][i] = ExpectedPass(st.queue, p)[i]
    /\ \A l \in Leaves :
           \A i \in DOMAIN st.arena1[l] :
               \/ st.arena1[l][i] = NoRow
               \/ st.arena1[l][i] = ExpectedLeaf(st.queue, l)[i]

CapacityExact ==
    /\ \A p \in st.liveA0 :
           Len(st.arena0[p]) = Len(ExpectedPass(st.queue, p))
    /\ \A l \in st.liveA1 :
           Len(st.arena1[l]) = Len(ExpectedLeaf(st.queue, l))

StableAtBarriers ==
    /\ st.phase \in {"refalloc", "refine", "probe", "cancelProbe"}
       => \A p \in Passes : st.arena0[p] = ExpectedPass(st.queue, p)
    /\ st.phase \in {"probe", "cancelProbe"}
       => \A l \in Leaves : st.arena1[l] = ExpectedLeaf(st.queue, l)

WorkerBound == Cardinality(st.running) <= WORKERS

RaceFree ==
    \A j, k \in st.running :
        j # k => Footprint(j) \cap Footprint(k) = {}

ProbeOrder ==
    /\ st.nextEmit <= st.nextAdmit
    /\ \A l \in Leaves :
           st.probeState[l] = "emitted" => l < st.nextEmit
    /\ \A l \in {x \in Leaves : x < st.nextEmit} :
           st.probeState[l] = "emitted"
    /\ Cardinality(Outstanding) <= WORKERS
    /\ st.liveResults = {l \in Leaves : st.probeState[l] = "ok"}

MemoryBound ==
    /\ ~(st.liveEntries # {} /\ st.liveA1 # {})
    /\ ~(st.liveEntries # {} /\ st.liveResults # {})
    /\ ~(st.liveA0 # {} /\ st.liveResults # {})
    /\ Cardinality(st.liveResults) <= WORKERS

OutputRefinement ==
    st.output = WavesOutput(st.doneWaves)
                \o LeavesOutput(st.queue, st.nextEmit)

FailureSafety ==
    /\ st.phase = "failed" => st.running = {}
    /\ st.phase = "failed" =>
           /\ st.liveEntries = {}
           /\ st.liveA0 = {}
           /\ st.liveA1 = {}
           /\ st.liveResults = {}
    /\ st.primary = PreError => Len(st.output) = st.waveOutputPrefix
    /\ \A l \in Leaves :
           st.primary = ErrorOf(l) => st.nextEmit = l

Safety ==
    /\ TypeOK
    /\ WaveExact
    /\ HashExactlyOnce
    /\ RankInjective
    /\ CellSafety
    /\ CapacityExact
    /\ StableAtBarriers
    /\ WorkerBound
    /\ RaceFree
    /\ ProbeOrder
    /\ MemoryBound
    /\ OutputRefinement
    /\ FailureSafety

FinalRefinement ==
    st.phase = "done" => st.output = WavesOutput(CanonicalWaves)

StopCondition == ~EnvironmentOK

Fairness ==
    /\ WF_vars(Accumulate)
    /\ WF_vars(CloseFinal)
    /\ WF_vars(FinishInput)
    /\ \A j \in AllJobs : WF_vars(StartJob(j))
    /\ \A p \in Passes : WF_vars(FinishPre(p))
    /\ WF_vars(PreBarrier)
    /\ \A o \in Occ : WF_vars(FinishScatter(o))
    /\ WF_vars(ScatterBarrier)
    /\ \A p \in Passes : WF_vars(FinishRefAlloc(p))
    /\ WF_vars(RefAllocBarrier)
    /\ \A o \in Occ : WF_vars(FinishRefine(o))
    /\ WF_vars(RefineBarrier)
    /\ \A j \in AllJobs : WF_vars(CancelNonProbe(j))
    /\ WF_vars(FinishNonProbeFailure)
    /\ \A l \in Leaves : WF_vars(StartProbe(l))
    /\ \A l \in Leaves : WF_vars(FinishProbe(l))
    /\ \A l \in Leaves : WF_vars(EmitLeaf(l))
    /\ \A l \in Leaves : WF_vars(ObserveProbeFailure(l))
    /\ \A l \in Leaves : WF_vars(CancelProbe(l))
    /\ WF_vars(FinishProbeFailure)
    /\ WF_vars(CompleteWave)

FairSpec == Spec /\ Fairness
Termination == <>(st.phase \in {"done", "failed"})

THEOREM SafetyTheorem == Spec => []Safety
THEOREM RefinementTheorem == Spec => []FinalRefinement
THEOREM TerminationTheorem == FairSpec => Termination

\* If implementation evidence makes StopCondition true: halt, name the
\* falsified conjunct, propose the minimum revision, and do not add fallback.

=============================================================================
