----------------------------- MODULE MC_PL1 -----------------------------
(* PL = 1: the refalloc/refine stages are skipped by the same machine; the  *)
(* pass arenas transfer to the leaf arenas at the scatter barrier.  Ends    *)
(* with EOF on an empty queue (no final partial wave).                      *)
EXTENDS WaveJoinProbe

mc_Input == <<<<0>>, <<1>>>>
mc_HashValue == 0 .. 1
mc_Bytes(b) == <<1, 1>>[b]
mc_Pass0(h) == h
mc_LeafOf(h) == 0

mc_Result == {"x1", "x2"}
mc_RowResult(o) ==
    CASE o = <<1, 1>> -> <<"x1">>
    []   o = <<2, 1>> -> <<"x2">>

mc_Error == {"efault"}
mc_ErrorOf(l) == "efault"
mc_FaultError == "efault"
mc_FailLeaves == {}
mc_FaultySteps == {}

=============================================================================
