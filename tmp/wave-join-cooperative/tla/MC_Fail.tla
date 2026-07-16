----------------------------- MODULE MC_Fail -----------------------------
(* Failure and cancellation: leaf 1's probe always faults (ErrorOf(1)),    *)
(* scan and pre steps may fault (FaultError), and external cooperative     *)
(* cancellation may arrive at any time.  Every behavior terminates in      *)
(* "failed" with first-exception-wins and exactly-once release.            *)
EXTENDS WaveJoinProbe

mc_Input == <<<<1, 3>>, <<0>>>>
mc_HashValue == 0 .. 3
mc_Bytes(b) == <<1, 1>>[b]
mc_Pass0(h) == h \div 2
mc_LeafOf(h) == h % 2

mc_Result == {"f11", "f12", "f21"}
mc_RowResult(o) ==
    CASE o = <<1, 1>> -> <<"f11">>
    []   o = <<1, 2>> -> <<"f12">>
    []   o = <<2, 1>> -> <<"f21">>

mc_Error == {"eL1", "eOther", "efault"}
mc_ErrorOf(l) == IF l = 1 THEN "eL1" ELSE "eOther"
mc_FaultError == "efault"
mc_FailLeaves == {1}
mc_FaultySteps == {"scan", "pre"}

TerminationFail == <>(st.phase = "failed")

=============================================================================
