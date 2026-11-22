# Why `T = T / s` in Amit’s `trans_cortex.m`?

## What the code says

The **only in-code explanation** is Amit’s comment in
`trans_cortex.m` line 35:

```matlab
T = T/s; % s is alpha - why did they devide T by alpha? According to M. Schira T = theta*alpha
```

So the author **does not justify the divide** — they flag that it **disagrees**
with Schira’s stated compression `P(θ) = θ·α`.

## What the rest of the lab tree documents

Everywhere else (comments only, not a separate write-up), Amit/Joanna scripts
**describe** Schira the paper way:

- `retinoForNoam.m`, `retino_trans_*Hemisphere.m`, parameter save scripts:
  `% s-alpha from P(theta)=theta*alpha`
- `retinotopic_model_joanna.m` (draft): `P = alpha*theta` then
  `fa = sech(P).^fa_exp` (multiply convention).

Joanna’s optimization header cites **Ayzenshtat et al.** and **Mark Schira** for
the *model family*, not for deriving `T/s`.

There is **no** `.m` file, README, or checked-in note that says “we divide
because …”. The Word manual (`JOANNA OPTIMIZATION MANUAL.docx`) was not
parsed here; if it mentions α, it would be the next place to check manually.

## Likely interpretation (inference, not lab-confirmed)

1. **Inherited Amit GUI code** — `trans_cortex` predates Joanna’s thesis draft;
   optimization was run against **divide + abs VF + folded `w` display**, so
   winners `(a, α, k)` are **tuned to that map**, not to `P=αθ` on signed VF.

2. **Parameter name collision** — `s` is documented as “alpha from
   `P=θ·α`” but applied as **divisor** on θ before `sech(T)^…`. That can be
   an old sign/convention bug, or an implicit redefinition of `s` (same symbol,
   opposite effect on angle).

3. **Commented but unused fix** — `logAngle = abs(angle(w)); %/s;` suggests
   someone considered dividing the **output** angle by `s` instead of input `T`;
   the `/s` on input was left as-is.

4. **No evidence they “meant” multiply** in the live pipeline — Joanna never
   switched `trans_cortex` to multiply; she re-fit `(a, α)` under divide.

## Practical takeaway for this project

To match **prior student overlays and Opt_results RMSD**, reproduce **divide +
abs linspace VF + trans_cortex OM**, then chamber registration — not the
paper/thesis multiply map alone.

If we later adopt multiply for encoding, treat `(a, α)` from legacy mats as
**not portable** without re-optimization.
