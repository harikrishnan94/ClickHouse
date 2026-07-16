---------------------------- MODULE MC_Normal ----------------------------
(* Normal run: PL > 1, one budget-sealed wave plus an EOF final partial     *)
(* wave, a multi-match row, a zero-match row and an empty leaf.             *)
EXTENDS WaveJoinProbe

mc_Input == <<<<0, 3>>, <<1, 0>>, <<2>>>>
mc_HashValue == 0 .. 3
mc_Bytes(b) == <<2, 2, 1>>[b]
mc_Pass0(h) == h \div 2
mc_LeafOf(h) == h % 2

mc_Result == {"r11", "r12a", "r12b", "r21", "r31"}
mc_RowResult(o) ==
    CASE o = <<1, 1>> -> <<"r11">>
    []   o = <<1, 2>> -> <<"r12a", "r12b">>
    []   o = <<2, 1>> -> <<"r21">>
    []   o = <<2, 2>> -> <<>>
    []   o = <<3, 1>> -> <<"r31">>

mc_Error == {"efault"}
mc_ErrorOf(l) == "efault"
mc_FaultError == "efault"
mc_FailLeaves == {}
mc_FaultySteps == {}

(* Reachability witnesses (checked in the MC_Reach* configurations, where a
   violation is the EXPECTED result): the cooperative states really occur. *)
NeverFullOwnership ==
    Cardinality({w \in WorkerIds : st.wk[w].job # NoJob}) < WORKERS

NeverTwoInflight ==
    Cardinality({w \in WorkerIds : st.wk[w].res # NoBlock}) < 2

NeverCrossWithInflight ==
    ~(st.crossed /\ \E w \in WorkerIds : st.wk[w].res # NoBlock)

=============================================================================
