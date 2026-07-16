--------------------------- MODULE MC_CancelRace ---------------------------
(* Cancellation racing NORMAL completion, under the same machine: no       *)
(* injected faults, but external cancellation may arrive at any point of a *)
(* run that would otherwise complete two waves (one budget-sealed, one EOF *)
(* partial) and reach "done".  This makes cancellation race Seal, every    *)
(* barrier, CompleteWave (probeDone = Leaves but not yet completed),       *)
(* EOFSeal, FinishInput, and the second wave's drain with doneWaves        *)
(* already non-empty — interleavings no fault-only configuration reaches.  *)
EXTENDS WaveJoinProbe

mc_Input == <<<<0>>, <<3>>, <<1>>>>
mc_HashValue == 0 .. 3
mc_Bytes(b) == <<1, 1, 1>>[b]
mc_Pass0(h) == h \div 2
mc_LeafOf(h) == h % 2

mc_Result == {"c1", "c2", "c3"}
mc_RowResult(o) ==
    CASE o = <<1, 1>> -> <<"c1">>
    []   o = <<2, 1>> -> <<"c2">>
    []   o = <<3, 1>> -> <<"c3">>

mc_Error == {"efault"}
mc_ErrorOf(l) == "efault"
mc_FaultError == "efault"
mc_FailLeaves == {}
mc_FaultySteps == {}

=============================================================================
