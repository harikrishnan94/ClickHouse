#!/bin/bash
# Cross-tree codegen parity runs for the loop groups that lacked codegen evidence (X1).
# Each invocation must resolve to exactly one baseline and one unified symbol; xtree.py
# exits 3 and lists the candidates if it does not.
set -u
cd "$(dirname "$0")"
O=codegen/xtree
mkdir -p "$O"

INS_BASE_2L='^void DB::HashJoinMethods<\(DB::JoinKind\)0, \(DB::JoinStrictness\)3, DB::HashJoin::MapsTemplate<DB::RowRefList>>::insertFromBlockImplTypeCase<DB::ColumnsHashing::HashMethodOneNumber<PairNoInit<unsigned long, DB::RowRefList>.*, TwoLevelHashMapTable<unsigned long,.*, DB::ColumnVector<unsigned long>>\('
INS_UNI_2L='^void DB::Unified::HashJoinMethods<\(DB::JoinKind\)0, \(DB::JoinStrictness\)3, DB::Unified::HashJoin::MapsTemplate<DB::RowRefList>>::insertFromBlockImplTypeCase<DB::ColumnsHashing::HashMethodOneNumber<PairNoInit<unsigned long, DB::RowRefList>.*, TwoLevelHashMapTable<unsigned long,.*, DB::ColumnVector<unsigned long>>\('
INS_BASE_FLAT='^void DB::HashJoinMethods<\(DB::JoinKind\)0, \(DB::JoinStrictness\)3, DB::HashJoin::MapsTemplate<DB::RowRefList>>::insertFromBlockImplTypeCase<DB::ColumnsHashing::HashMethodOneNumber<PairNoInit<unsigned long, DB::RowRefList>.*, HashMapTable<unsigned long, HashMapCell.*, DB::ColumnVector<unsigned long>>\('

run() { # label base_regex unified_regex out
  local label="$1" base="$2" uni="$3" out="$4"
  echo "### $label -> $O/$out.log"
  python3 xtree.py --label "$label" --base "$base" ${uni:+--unified "$uni"} \
      --examples 6 --dump "$O/$out.txt" > "$O/$out.log" 2>&1
  echo "    exit=$?"
}

run 'B11/B12 insert (two-level vs two-level)' "$INS_BASE_2L" "$INS_UNI_2L" B11_2L
run 'B11/B12 insert (baseline FLAT vs unified two-level)' "$INS_BASE_FLAT" "$INS_UNI_2L" B11_flat
run 'P6 addFoundRowAll' \
  '^void DB::addFoundRowAll<TwoLevelHashMapTable<unsigned long, HashMapCell<unsigned long, DB::RowRefList.*HashMapTable, 8>::mapped_type const&, DB::AddedColumns<false>&' \
  '^void DB::Unified::addFoundRowAll<TwoLevelHashMapTable<unsigned long, HashMapCell<unsigned long, DB::RowRefList.*HashMapTable, -1>::mapped_type const&, DB::Unified::AddedColumns<false>&' P6
run 'P8/P0 joinBlockImpl' \
  '^DB::HashJoinMethods<\(DB::JoinKind\)0, \(DB::JoinStrictness\)3, DB::HashJoin::MapsTemplate<DB::RowRefList>>::joinBlockImpl\(DB::HashJoin const&, DB::ScatteredBlock' \
  '^DB::Unified::HashJoinMethods<\(DB::JoinKind\)0, \(DB::JoinStrictness\)3, DB::Unified::HashJoin::MapsTemplate<DB::RowRefList>>::joinBlockImpl\(DB::Unified::HashJoin const&, DB::ScatteredBlock' P8_P0
run 'N1/N3/N4/N7 fillColumns' \
  '^unsigned long DB::NotJoinedHash::fillColumns<TwoLevelHashMapTable<unsigned long, HashMapCell<unsigned long, DB::RowRefList' \
  '^unsigned long DB::Unified::NotJoinedHash::fillColumns<TwoLevelHashMapTable<unsigned long, HashMapCell<unsigned long, DB::RowRefList' N1_N7
run 'N6 fillNullsFromBlocks' '^DB::NotJoinedHash::fillNullsFromBlocks' '^DB::Unified::NotJoinedHash::fillNullsFromBlocks' N6
run 'B2/B3/B4 addBlockToJoin' \
  '^DB::HashJoin::addBlockToJoin\(DB::Block const&, DB::detail::Selector, bool\)$' \
  '^DB::Unified::HashJoin::addBlockToJoin\(DB::Block const&, DB::detail::Selector, bool\)$' B234
run 'B2/B3/B4 addBlockToJoin (2-arg entry overload)' \
  '^DB::HashJoin::addBlockToJoin\(DB::Block const&, bool\)$' \
  '^DB::Unified::HashJoin::addBlockToJoin\(DB::Block const&, bool\)$' B234_entry
