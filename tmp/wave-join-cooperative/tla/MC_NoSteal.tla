---------------------------- MODULE MC_NoSteal ----------------------------
(* NEGATIVE WITNESS (expected to FAIL model checking).                      *)
(*                                                                          *)
(* Breaks the participation contract with the classic anti-pattern the     *)
(* cooperative design forbids: worker 0 is a dedicated scanner that never   *)
(* claims drain work, and leaf 0 is affine to worker 0 (static partition    *)
(* assignment instead of cooperative claiming).  Leaf 0's probe job then    *)
(* stays claimable forever while worker 1 idles: `ParticipationLive` must   *)
(* produce a counterexample.  This demonstrates the property is not a       *)
(* tautology: it can fail, and it fails exactly on dedicated-crew designs.  *)
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

broken_ClaimEligible(w, kind, id) ==
    /\ w # 0                            \* worker 0: dedicated scanner crew
    /\ ~(kind = "probe" /\ id = 0)      \* leaf 0: affine to worker 0 only

=============================================================================
