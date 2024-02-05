#set heading(numbering: "1.1.")
#set text(font: "New Computer Modern")
#set math.equation(numbering: "(1)")
#show ref: it => {
  let eq = math.equation
  let el = it.element
  if el != none and el.func() == eq {
    // Override equation references.
    numbering(
      el.numbering,
      ..counter(eq).at(el.location())
    )
  } else {
    // Other references as usual.
    it
  }
}

#let title = "Stochastic Equilibrium Model of Peak Period Traffic Congestion"

#align(center, text(17pt)[*#title*])

= Model

*Bottleneck*

There is a bottleneck with capacity $s$.
The travel time from origin to destination is
$ "tt"(t) = D(t) / s, $ <travel_time>
where $t$ is the departure time from origin and $D(t)$ is the number of cars in the queue at time $t$.

Denote by $t_q$ and $t_(q')$ the beginning and the end times of the congestion period, then
$ dot(D)(t) = 0, quad "for" t ≤ t_q "and" t ≥ t_(q'), $ <queue_rate0>
$ dot(D)(t) = r(t) - s, quad "for" t_q < t < t_(q'), $ <queue_rate1>
where $r(t)$ is the departure rate from origin at time $t$.

*Departure-time choice model*

The unobservable random utility of departing from origin at time $t$ is
$ U(t) = V(t) + μ ε(t), $
where $V(t)$ is a deterministic utility component, $ε(t)$ is a disturbance term and $μ$ is a scale parameter.

Using a continuous logit model, the probability of a unit time interval at $t$ being chosen is given by
$ p(t) = exp[V(t) / μ] / E, $
where
$ E = integral_(0)^T exp[V(u) / μ] dif u, $
with the possible departure time window being $[0, T]$.

The departure rate from origin, $r(t)$, can thus be written as
$ r(t) = N exp[V(t) / μ]/E, $ <dep_rate>
where $N$ is the number of drivers.

*Utility function*

The deterministic utility component is
$ V(t) = β θ(t) t + (β θ(t) - α) "tt"(t) + β Δ |θ(t)| -β θ(t) t^*, $ <utility>
with
- $α$ the marginal disutility of an additional unit of travel time,
- $[t^* - Δ, t^* + Δ]$ the desired time period for arrival at destination,
- $β$ is the marginal disutility of arriving early by an additional unit of time,
- $γ$ is the marginal disutility of arriving late by an additional unit of time,
- $ θ(t) = cases(
    1 "for" 0 ≤ t ≤ tilde(t)\,,
    0 "for" tilde(t) < t < hat(t)\,,
    -γ/β "for" hat(t) ≤ t ≤ T.,
  ) $
- $tilde(t) = t^* - Δ - "tt"(tilde(t))$ is the earliest on-time departure time,
- $hat(t) = t^* + Δ - "tt"(hat(t))$ is the latest on-time departure time.

*Equilibrium*

The congestion states begins at time $t_q$ and ends at time $t_(q')$.
// This means that
// $ D(t) = 0, quad "for" 0 ≤ t ≤ t_q "and" t_(q') ≤ t ≤ T, $
// and
// $ D(t) > 0, quad "for" t_q < t < t_(q'). $
Solving the differential equation defined by @queue_rate0 and @queue_rate1 gives
$ D(t) = 0, quad "for" 0 ≤ t ≤ t_q "and" t_(q') ≤ t ≤ T, $
$ D(t) = integral_(t_q)^t r(t) - (t - t_q) s, quad "for" t_q < t < t_(q'). $
Using equation @dep_rate:
$ D(t) = integral_(t_q)^t r(u) dif u - (t - t_q) s, quad "for" t_q < t < t_(q'). $ <queue_rate2>
Substituting @queue_rate2 in @travel_time yields
$ "tt"(t) = (1/s) integral_(t_q)^t r(u) dif u - (t - t_q), quad "for" t_q < t < t_(q'). $ <travel_time_eq>
Using @travel_time_eq in @utility, the departure rate is @dep_rate:
$ r(t) = A_q (θ(t)) exp[((β θ(t) - α) / (μ s)) integral_(t_q)^t r(u) dif u + (α t) / μ], quad "for" t_q < t < t_q', $ <dep_rate_implicit>
with
$ A_q (θ) = (N/E) exp[(β Δ |θ| - β θ t^* + (β θ - α) t_q) / μ ]. $
We differentiate equation @dep_rate_implicit with respect to $t$:
$ dot(r)(t) = ((β θ(t) - α) / (μ s)) r^2(t) + (α / μ) r(t), quad "for" t_q < t < t_q'. $ <dep_rate_diff>
A solution of $dot(r)(t) = a r(t) + b(t) r^2(t)$ is $r(t) = - e^(a t) / (integral b(u) e^(a u) dif u)$ (Bernoulli differential equation).
With some computation, I found that a solution of @dep_rate_diff is
$ r(t) = ( α s exp[(α slash μ) t] ) / ((α - β θ(t)) exp[(α slash μ) t] + β integral dot(θ)(u) exp[(α slash μ) u] dif u), quad "for" t_q < t < t_q'. $
The solution from the paper:
$ r(t) = (α s exp[(α slash μ) t]) / ((α - β θ(t)) exp[(α slash μ) t] + α s K(θ(t))), quad "for" t_q < t < t_q' $ <dep_rate_solution>
$ K(θ) = [1 / r(t_l) - (α - β θ) / (α s)] exp[(α slash μ) t_l] $
$ t_l = cases(
  t_q "for" t_q ≤ t ≤ tilde(t),
  tilde(t) "for" tilde(t) < t < hat(t),
  hat(t) "for" hat(t) ≤ t ≤ t_q',
) $

The two solutions coincide if:
$ β integral dot(θ)(u) exp[(α slash μ) u] dif u &= α s K(θ(t)) \
  &= [(α s) / r(t_l) - (α - β θ(t))] exp[(α slash μ) t_l]
$

$ θ(t) = cases(
  1 "for" 0 ≤ t ≤ tilde(t)\,,
  0 "for" tilde(t) < t < hat(t)\,,
  -γ slash β "for" hat(t) ≤ t ≤ T.,
) $

The solution for the queue length is obtained by solving equation @queue_rate1:
$ D(t) = cases(
  integral_(t_q)^t r(u) dif u - s (t - t_q) "for" t_q ≤ t ≤ tilde(t),
  D(tilde(t)) + integral_(tilde(t))^t r(u) dif u - s (t - tilde(t)) "for" tilde(t) < t ≤ hat(t),
  D(hat(t)) + integral_(hat(t))^t r(u) dif u - s (t - hat(t)) "for" hat(t) < t ≤ t_(q'),
) $

We have:
$ K(1) = [1 / r(t_q) - (α - β) / (α s)] exp[(α slash μ) t_q] $
$ K(0) = [1 / r(tilde(t)) - 1 / s] exp[(α slash μ) tilde(t)] $
$ K(-γ slash β) = [1 / r(hat(t)) - (α + γ) / (α s)] exp[(α slash μ) hat(t)] $

Assuming that $t_q < tilde(t)$:
$ V(t_q) = β (t_q + Δ - t^*) $
$ r(t_q) = N/E exp[(β slash μ) (t_q + Δ - t^*)] $

Assuming that $t_q' > hat(t)$:
$ V(t_q') = γ (t^* + Δ - t_q') $
$ r(t_q') = N/E exp[(γ slash μ) (t^* + Δ - t_q')] $

$ integral_(t_l)^t r(u) dif u &= α s integral_(t_l)^t exp[(α slash μ) t] / ((α - β θ(t)) exp[(α slash μ) t] + α s K(θ(t))) dif u \
  &= (μ s) / (α - β θ(t))  [ ln[(α - β θ(t)) exp[(α slash μ) t] + α s K(θ(t))]]_(t_l)^t \
  &= (μ s) / (α - β θ(t))  ln[ ((α - β θ(t)) exp[(α slash μ) t] + α s K(θ(t))) / ((α - β θ(t_l)) exp[(α slash μ) t_l] + α s K(θ(t_l))) ] \
  &= (μ s) / (α - β θ(t))  ln[ (exp[(α slash μ) t] / r(t)) / (exp[(α slash μ) t_l] / r(t_l)) ] \
  &= (μ s) / (α - β θ(t)) [ α / μ (t - t_l) - ln( r(t) / r(t_l) ) ] \
  &= (α s) / (α - β θ(t)) (t - t_l) - (μ s) / (α - β θ(t)) ln( r(t) / r(t_l) ) $

$ integral_(t_l)^t r(u) dif u - s (t - t_l) &= [(α s) / (α - β θ(t)) - s] (t - t_l) - (μ s) / (α - β θ(t)) ln( r(t) / r(t_l) ) \
  &= s / (α - β θ(t)) [β θ(t) (t - t_l) - μ ln( r(t) / r(t_l) ) ] $

Using equation @dep_rate_solution:
$ D(t) = cases(
  integral_(t_q)^t r(u) dif u - s (t - t_q) "for" t_q ≤ t ≤ tilde(t),
  D(tilde(t)) + integral_(tilde(t))^t r(u) dif u - s (t - tilde(t)) "for" tilde(t) < t ≤ hat(t),
  D(hat(t)) + integral_(hat(t))^t r(u) dif u - s (t - hat(t)) "for" hat(t) < t ≤ t_(q'),
) $

$ D(t) = cases(
  s / (α - β) [β (t - t_q) - μ ln( r(t) / r(t_q) ) ] "for" t_q ≤ t ≤ tilde(t),
  D(tilde(t)) - (μ s) / α ln( r(t) / r(tilde(t)) ) "for" tilde(t) < t ≤ hat(t),
  D(hat(t)) - s / (α + γ) [γ (t - hat(t)) + μ ln( r(t) / r(hat(t)) ) ] "for" hat(t) < t ≤ t_(q'),
) $

*Final solution*

$ r(t) = cases(
  N / E exp[(β / μ) (t + Δ - t^*)] "for" 0 ≤ t ≤ t_q,
  [(α - β) / (α s) + [1/r(t_q) - (α - β) / (α s)] exp[-(α / μ) (t - t_q)]]^(-1) "for" t_q < t ≤ tilde(t),
  [1/s + [1 / r(tilde(t)) - 1/s] exp[-(α / μ) (t - tilde(t))]]^(-1) "for" tilde(t) < t ≤ hat(t),
  [(α + γ) / (α s) + [1/r(hat(t)) - (α + γ) / (α s)] exp[-(α / μ) (t - hat(t))]]^(-1) "for" hat(t) < t ≤ t_q',
  N / E exp[(γ / μ) (t^* + Δ - t)] "for" t_q' < t ≤ T,
) $

$ D(t) = cases(
  0 "for" 0 ≤ t ≤ t_q,
  s / (α - β) [β (t - t_q) - μ ln( r(t) / r(t_q) ) ] "for" t_q < t ≤ tilde(t),
  D(tilde(t)) - (μ s) / α ln( r(t) / r(tilde(t)) ) "for" tilde(t) < t ≤ hat(t),
  D(hat(t)) - s / (α + γ) [γ (t - hat(t)) + μ ln( r(t) / r(hat(t)) ) ] "for" hat(t) < t ≤ t_(q'),
  0 "for" t_q' < t ≤ T,
) $

$ tilde(t) = t^* - Δ - D(tilde(t)) / s $

$ tilde(t) = t^* - Δ - (β (tilde(t) - t_q) - μ ln( r(tilde(t)) / r(t_q))) / (α - β) $

For $t_q < t ≤ tilde(t)$:
$ μ ln(r(t) / r(t_q)) = α t - μ ln( (α - β) / (α s) exp[(α slash μ) t] + K(1) ) - μ ln(r(t_q)) $
$ μ ln(r(t) / r(t_q)) = α t - μ ln( (α - β) / (α s) exp[(α slash μ) t] r(t_q) + K(1) r(t_q) ) $
$ μ ln(r(t) / r(t_q)) = α t + μ ln( (α s) / (α - β)) - μ ln( exp[(α slash μ) t] r(t_q) + ((α s) / (α - β) - r(t_q)) exp[(α slash μ) t_q] ) $
$ μ ln(r(t) / r(t_q)) = μ ln( (α s) / (α - β)) - μ ln( r(t_q) + ((α s) / (α - β) - r(t_q)) exp[-(α slash μ) (t - t_q)] ) $

$ μ ln( r(hat(t)) / r(tilde(t)) ) =& μ ln(α s) + α hat(t) - μ ln{α exp[(α slash μ) hat(t)] + α s K(0)} \
  & - μ ln(α s) - α tilde(t) + μ ln{ (α - β) exp[(α slash μ) tilde(t)] + α s K(1)} $
$ μ ln( r(hat(t)) / r(tilde(t)) ) =& α (hat(t) - tilde(t)) - μ ln{α exp[(α slash μ) hat(t)] + α s [ 1 / r(tilde(t)) - 1 / s] exp[(α slash μ) tilde(t)]} \
  & + μ ln{ (α - β) exp[(α slash μ) tilde(t)] + α s [ 1 / r(t_q) - (α - β) / (α s)] exp[(α slash μ) t_q]} $

$ integral_(t_q)^(t_q') r(u) dif u =& s { α / (α - β) (tilde(t) - t_q) + (hat(t) - tilde(t)) + α / (α + γ) (t_q' - hat(t)) \
    &- μ [ 1 / (α - β) ln(r(tilde(t)) / r(t_q)) + 1 / α ln(r(hat(t)) / r(tilde(t))) + 1 / (α + γ) ln(r(t_q') / r(hat(t))) ] } $

= Simplifications I make compared to the original version

- $t_0 = t_1 = t_2 = 0$ (travel time from origin to bottleneck and from bottleneck to destination is null)
- $t_u = 0$ (not considering the extended bottleneck model)
