"""
Cross-venue maker/hedge backtest.

Structure, mirroring the FX strategy: rest a passive quote on the WIDER book,
and when it fills, immediately cross the TIGHTER book to flatten.  The position
is flat within one latency hop, so the P&L of a round trip is locked at hedge
time and nothing depends on a holding-period assumption:

    pnl_per_coin = hedge_price - fill_price          (for a passive BUY)
                 = [ mid_H(t_h) - mid_M(t_f) ]       basis + drift over the hop
                 + [ mid_M(t_f) - Q ]                half-spread earned
                 - [ mid_H(t_h) - bid_H(t_h) ]       half-spread paid
                 - fees

WHAT MAKES THIS DECIDABLE, unlike the Cboe work: the fill is not modelled.  The
aggTrades tape gives every aggressive trade with its side, so a resting bid at
Q is filled exactly when enough sell-aggressor volume prints at <= Q to clear
the queue that was ahead of it.  Queue position is taken from the displayed
size at the moment the order is placed, and the order goes to the BACK of it.

Everything is charged: exchange fees on both legs, latency on the hedge, and
the order is cancelled and re-queued whenever the touch moves.
"""
import numpy as np
from numba import njit

import load


@njit(cache=True)
def _simulate(bts, bbid, bbq, bask, baq,          # maker venue book
              tts, tpx, tqty, tagg,               # maker venue trades
              hts, hbid, hask,                    # hedge venue book
              clip, lat_ms, join_only, max_pos, queue_mult):
    """
    Walk the maker venue's book and tape together.  Returns one row per fill:
      side(+1 passive buy / -1 passive sell), fill_px, hedge_px,
      maker_mid_at_fill, hedge_mid_at_hedge, queue_ahead_at_place, ts
    """
    nb = bts.shape[0]
    nt = tts.shape[0]
    nh = hts.shape[0]

    out_side = np.empty(nt, np.float64)
    out_fill = np.empty(nt, np.float64)
    out_hedge = np.empty(nt, np.float64)
    out_mmid = np.empty(nt, np.float64)
    out_hmid = np.empty(nt, np.float64)
    out_q = np.empty(nt, np.float64)
    out_ts = np.empty(nt, np.int64)
    m = 0

    ib = 0          # book cursor
    ih = 0          # hedge cursor
    # our two resting orders: price, remaining size, queue ahead
    bid_px = 0.0
    bid_q = 0.0
    bid_ahead = 0.0
    ask_px = 0.0
    ask_q = 0.0
    ask_ahead = 0.0
    pos = 0.0

    for it in range(nt):
        t = tts[it]
        # advance the maker book to this instant
        while ib + 1 < nb and bts[ib + 1] <= t:
            ib += 1
        cb = bbid[ib]
        ca = bask[ib]
        if cb <= 0.0 or ca <= cb:
            continue

        # (re)place quotes at the touch; a moved touch means a new queue
        if bid_px != cb:
            bid_px = cb
            bid_q = clip
            bid_ahead = bbq[ib] * queue_mult
        if ask_px != ca:
            ask_px = ca
            ask_q = clip
            ask_ahead = baq[ib] * queue_mult

        px = tpx[it]
        qty = tqty[it]
        agg = tagg[it]

        # --- does this aggressive print reach our resting order?
        filled = 0.0
        side = 0.0
        fpx = 0.0
        if agg < 0.0 and bid_q > 0.0 and px <= bid_px and pos < max_pos:
            # market sell into the bid side: consume the queue ahead of us first
            v = qty
            if bid_ahead > 0.0:
                use = v if v < bid_ahead else bid_ahead
                bid_ahead -= use
                v -= use
            if v > 0.0 and bid_ahead <= 0.0:
                take = v if v < bid_q else bid_q
                bid_q -= take
                filled = take
                side = 1.0
                fpx = bid_px
        elif agg > 0.0 and ask_q > 0.0 and px >= ask_px and pos > -max_pos:
            v = qty
            if ask_ahead > 0.0:
                use = v if v < ask_ahead else ask_ahead
                ask_ahead -= use
                v -= use
            if v > 0.0 and ask_ahead <= 0.0:
                take = v if v < ask_q else ask_q
                ask_q -= take
                filled = take
                side = -1.0
                fpx = ask_px

        if filled <= 0.0:
            continue
        if join_only and filled < clip * 1e-9:
            continue

        # --- hedge on the other venue, one latency hop later
        th = t + lat_ms
        while ih + 1 < nh and hts[ih + 1] <= th:
            ih += 1
        hb = hbid[ih]
        ha = hask[ih]
        if hb <= 0.0 or ha <= hb:
            continue
        hpx = hb if side > 0.0 else ha          # sell into bid / buy the ask

        out_side[m] = side
        out_fill[m] = fpx
        out_hedge[m] = hpx
        out_mmid[m] = 0.5 * (cb + ca)
        out_hmid[m] = 0.5 * (hb + ha)
        out_q[m] = filled
        out_ts[m] = t
        m += 1
        pos += side * filled
        # the hedge flattens us again
        pos -= side * filled

    return (out_side[:m], out_fill[:m], out_hedge[:m], out_mmid[:m],
            out_hmid[:m], out_q[:m], out_ts[:m])


def run(maker_sym, hedge_sym, day, clip, lat_ms=5, max_pos=1e18,
        queue_mult=1.0):
    mb = load.book(maker_sym, day)
    mt = load.trades(maker_sym, day)
    hb = load.book(hedge_sym, day)
    if mb is None or mt is None or hb is None:
        return None
    side, fill, hedge, mmid, hmid, q, ts = _simulate(
        mb[0], mb[1], mb[2], mb[3], mb[4],
        mt[0], mt[1], mt[2], mt[3],
        hb[0], hb[1], hb[3],
        float(clip), int(lat_ms), False, float(max_pos),
        float(queue_mult))
    return dict(side=side, fill=fill, hedge=hedge, mmid=mmid, hmid=hmid,
                qty=q, ts=ts, maker=maker_sym, hedge_sym=hedge_sym, day=day)


def pnl(res, maker_fee_bp, taker_fee_bp):
    """
    Per-round-trip economics in basis points of notional, plus the decomposition.
    Sizes are converted to coin units so the two venues are comparable.
    """
    s, f, h = res["side"], res["fill"], res["hedge"]
    mm, hm, q = res["mmid"], res["hmid"], res["qty"]
    coin = load.to_base_units(res["maker"], q, f)

    gross_bp = s * (h - f) / f * 1e4
    half_earned = s * (mm - f) / f * 1e4          # what our quote earned vs mid
    basis_drift = s * (hm - mm) / f * 1e4         # basis + drift over the hop
    half_paid = -np.abs(hm - h) / f * 1e4         # crossing the hedge venue
    fees_bp = maker_fee_bp + taker_fee_bp
    net_bp = gross_bp - fees_bp

    notional = coin * f
    return dict(
        n=len(f),
        coin=float(coin.sum()),
        notional=float(notional.sum()),
        gross_bp=float(np.average(gross_bp, weights=notional)) if len(f) else 0.0,
        half_earned_bp=float(np.average(half_earned, weights=notional)) if len(f) else 0.0,
        basis_drift_bp=float(np.average(basis_drift, weights=notional)) if len(f) else 0.0,
        half_paid_bp=float(np.average(half_paid, weights=notional)) if len(f) else 0.0,
        fees_bp=fees_bp,
        net_bp=float(np.average(net_bp, weights=notional)) if len(f) else 0.0,
        net_usd=float((net_bp / 1e4 * notional).sum()),
        gross_usd=float((gross_bp / 1e4 * notional).sum()),
    )
