--------------------------- MODULE MC_MultiWave ---------------------------
(* Two budget-sealed waves with three workers.  Two rows in DIFFERENT waves *)
(* produce the same result value ("dup"): the final refinement must count   *)
(* it twice, so any set-semantics collapse in the output accounting fails.  *)
EXTENDS WaveJoinProbe

mc_Input == <<<<0>>, <<3>>, <<0>>, <<2>>>>
mc_HashValue == 0 .. 3
mc_Bytes(b) == <<1, 1, 1, 1>>[b]
mc_Pass0(h) == h \div 2
mc_LeafOf(h) == h % 2

mc_Result == {"dup", "m2", "m4"}
mc_RowResult(o) ==
    CASE o = <<1, 1>> -> <<"dup">>
    []   o = <<2, 1>> -> <<"m2">>
    []   o = <<3, 1>> -> <<"dup">>
    []   o = <<4, 1>> -> <<"m4">>

mc_Error == {"efault"}
mc_ErrorOf(l) == "efault"
mc_FaultError == "efault"
mc_FailLeaves == {}
mc_FaultySteps == {}

=============================================================================
