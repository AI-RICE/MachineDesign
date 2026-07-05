#!/bin/bash
# read-seats.sh — print live ANSYS Electronics license seat counts from the shared campus
# FlexLM server (the pool a Maxwell 2D solve draws from). Read-only; no root needed.
# Usage: bash read-seats.sh
LMUTIL=/data/Ansys/v242/licensingclient/linx64/lmutil
INI=/data/Ansys/shared_files/licensing/ansyslmd.ini
SRV=$(awk -F= '/SERVER/{print $2}' "$INI" 2>/dev/null | head -1)
[ -z "$SRV" ] && { echo "no SERVER line in $INI"; exit 1; }
echo "license server: $SRV"
# Features a 1-core Maxwell 2D transient solve checks out (base seats, NOT the contended
# anshpc parallel pool). "issued" = total seats; "in use" = currently checked out campus-wide.
"$LMUTIL" lmstat -a -c "$SRV" 2>&1 \
  | grep -E "Users of (electronics_desktop|elec_solve_2d|elec_solve_maxwell|anshpc):" \
  | sed -E 's/Users of /  /; s/;.*of /  in_use=/; s/ licenses? in use.*//; s/ *\(Total of /  seats=/'
echo "note: base seats (electronics_desktop/elec_solve_2d) are ~275, abundant; anshpc is the"
echo "      contended parallel pool -> stay 1-core per solve to avoid it."
