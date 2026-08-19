#!/usr/bin/env python3
"""Uncommitted higher-order phaser representation experiments.

This is a research cockpit, not a qualification harness.  It studies the
real-pole all-pass tail in dimensionless coordinates and compares ordinary
partial fractions with a whole-cluster Newton/product-rule image.
"""

from __future__ import annotations

import cmath
import hashlib
import math
import random
import struct
from dataclasses import dataclass

import mpmath as mp


MP_DPS = 100
SAMPLE_RATE = 44100.0
MATERIALIZER_TAIL_CUTOFF = 1.0e-12

# Correct order-16 CRAM partial fractions from Pusa (2012), arXiv:1206.2880.
# The approximation is alpha0 + 2*Re(sum alpha/(z-theta)).
CRAM16_THETA = [
    complex(-10.843917078696988026, 19.277446167181652284),
    complex(-5.2649713434426468895, 16.220221473167927305),
    complex(5.9481522689511774808, 3.5874573620183222829),
    complex(3.5091036084149180974, 8.4361989858843750826),
    complex(6.4161776990994341923, 1.1941223933701386874),
    complex(1.4193758971856659786, 10.925363484496722585),
    complex(4.9931747377179963991, 5.9968817136039422260),
    complex(-1.4139284624888862114, 13.497725698892745389),
]
CRAM16_ALPHA = [
    complex(-5.0901521865224915650e-7, -2.4220017652852287970e-5),
    complex(2.1151742182466030907e-4, 4.3892969647380673918e-3),
    complex(1.1339775178483930527e2, 1.0194721704215856450e2),
    complex(1.5059585270023467528e1, -5.7514052776421819979),
    complex(-6.4500878025539646595e1, -2.2459440762652096056e2),
    complex(-1.4793007113557999718, 1.7686588323782937906),
    complex(-6.2518392463207918892e1, -1.1190391094283228480e1),
    complex(4.1023136835410021273e-2, -1.5743466173455468191e-1),
]
CRAM16_ALPHA0 = 2.1248537104952237488e-16

# Order-48 incomplete-partial-factorization coefficients from Pusa (2016),
# as transcribed by OpenMC's depletion solver.  Unlike the order-16 PFD above,
# these factors update the real vector sequentially and scale by alpha0 last.
CRAM48_THETA = [
    complex(real, imag) for real, imag in zip(
        [-4.465731934165702e1, -5.284616241568964, -8.867715667624458,
         3.493013124279215, 1.564102508858634e1, 1.742097597385893e1,
         -2.834466755180654e1, 1.661569367939544e1, 8.011836167974721,
         -2.056267541998229, 1.449208170441839e1, 1.853807176907916e1,
         9.932562704505182, -2.244223871767187e1, 8.590014121680897e-1,
         -1.286192925744479e1, 1.164596909542055e1, 1.806076684783089e1,
         5.870672154659249, -3.542938819659747e1, 1.901323489060250e1,
         1.885508331552577e1, -1.734689708174982e1, 1.316284237125190e1],
        [6.233225190695437e1, 4.057499381311059e1, 4.325515754166724e1,
         3.281615453173585e1, 1.558061616372237e1, 1.076629305714420e1,
         5.492841024648724e1, 1.316994930024688e1, 2.780232111309410e1,
         3.794824788914354e1, 1.799988210051809e1, 5.974332563100539,
         2.532823409972962e1, 5.179633600312162e1, 3.536456194294350e1,
         4.600304902833652e1, 2.287153304140217e1, 8.368200580099821,
         3.029700159040121e1, 5.834381701800013e1, 1.194282058271408,
         3.583428564427879, 4.883941101108207e1, 2.042951874827759e1])
]
CRAM48_ALPHA = [
    complex(real, imag) for real, imag in zip(
        [6.387380733878774e2, 1.909896179065730e2, 4.236195226571914e2,
         4.645770595258726e2, 7.765163276752433e2, 1.907115136768522e3,
         2.909892685603256e3, 1.944772206620450e2, 1.382799786972332e5,
         5.628442079602433e3, 2.151681283794220e2, 1.324720240514420e3,
         1.617548476343347e4, 1.112729040439685e2, 1.074624783191125e2,
         8.835727765158191e1, 9.354078136054179e1, 9.418142823531573e1,
         1.040012390717851e2, 6.861882624343235e1, 8.766654491283722e1,
         1.056007619389650e2, 7.738987569039419e1, 1.041366366475571e2],
        [-6.743912502859256e2, -3.973203432721332e2, -2.041233768918671e3,
         -1.652917287299683e3, -1.783617639907328e4, -5.887068595142284e4,
         -9.953255345514560e3, -1.427131226068449e3, -3.256885197214938e6,
         -2.924284515884309e4, -1.121774011188224e3, -6.370088443140973e4,
         -1.008798413156542e6, -8.837109731680418e1, -1.457246116408180e2,
         -6.388286188419360e1, -2.195424319460237e2, -6.719055740098035e2,
         -1.693747595553868e2, -1.177598523430493e1, -4.596464999363902e3,
         -1.738294585524067e3, -4.311715386228984e1, -2.777743732451969e2])
]
CRAM48_ALPHA0 = 2.258038182743983e-47


def ratios(count: int) -> list[float]:
    return [2.0 ** (-1.25 + 2.5 * i / (count - 1))
            for i in range(count)]


def source_bank(partials: int, alpha: float) -> tuple[list[complex], list[complex]]:
    poles: list[complex] = []
    amps: list[complex] = []
    for index in range(partials):
        harmonic = index + 1
        sigma = 4.0 * (1.0 + 0.4 * harmonic)
        omega = 2.0 * math.pi * 220.0 * harmonic
        poles.append(complex(-sigma, omega) / alpha)
        amps.append(complex(harmonic ** -1.1, 0.0))
    return poles, amps


def product(values):
    result = 1
    for value in values:
        result *= value
    return result


@dataclass
class Image:
    nodes: list[mp.mpf]
    source_poles: list[mp.mpc]
    source_amps: list[mp.mpc]
    wet_source_amps: list[mp.mpc]
    tail_residues: list[mp.mpc]
    newton_coefficients: list[mp.mpc]
    scaled_weights: list[mp.mpc]


@dataclass(frozen=True)
class DD:
    """Two-word float64 used only by the off-audio materializer experiment."""
    hi: float
    lo: float = 0.0


@dataclass(frozen=True)
class ComplexDD:
    real: DD
    imag: DD


DD_SPLITTER = 134217729.0


def dd_two_sum(left: float, right: float) -> DD:
    total = left + right
    right_virtual = total - left
    error = (left - (total - right_virtual)) + (right - right_virtual)
    return DD(total, error)


def dd_quick_two_sum(left: float, right: float) -> DD:
    total = left + right
    return DD(total, right - (total - left))


def dd_two_product(left: float, right: float) -> DD:
    value = left * right
    left_split = DD_SPLITTER * left
    left_hi = left_split - (left_split - left)
    left_lo = left - left_hi
    right_split = DD_SPLITTER * right
    right_hi = right_split - (right_split - right)
    right_lo = right - right_hi
    error = ((left_hi * right_hi - value) + left_hi * right_lo
             + left_lo * right_hi) + left_lo * right_lo
    return DD(value, error)


def dd_add(left: DD, right: DD) -> DD:
    pair = dd_two_sum(left.hi, right.hi)
    return dd_quick_two_sum(pair.hi, pair.lo + left.lo + right.lo)


def dd_neg(value: DD) -> DD:
    return DD(-value.hi, -value.lo)


def dd_sub(left: DD, right: DD) -> DD:
    return dd_add(left, dd_neg(right))


def dd_mul(left: DD, right: DD) -> DD:
    pair = dd_two_product(left.hi, right.hi)
    error = pair.lo + left.hi * right.lo + left.lo * right.hi
    return dd_quick_two_sum(pair.hi, error)


def dd_div(left: DD, right: DD) -> DD:
    quotient_hi = left.hi / right.hi
    remainder = dd_sub(left, dd_mul(right, DD(quotient_hi)))
    quotient_lo = (remainder.hi + remainder.lo) / right.hi
    return dd_add(DD(quotient_hi), DD(quotient_lo))


def dd_value(value: DD) -> float:
    return value.hi + value.lo


def cdd(value: complex) -> ComplexDD:
    return ComplexDD(DD(value.real), DD(value.imag))


def cdd_add(left: ComplexDD, right: ComplexDD) -> ComplexDD:
    return ComplexDD(dd_add(left.real, right.real),
                     dd_add(left.imag, right.imag))


def cdd_sub(left: ComplexDD, right: ComplexDD) -> ComplexDD:
    return ComplexDD(dd_sub(left.real, right.real),
                     dd_sub(left.imag, right.imag))


def cdd_mul(left: ComplexDD, right: ComplexDD) -> ComplexDD:
    return ComplexDD(
        dd_sub(dd_mul(left.real, right.real),
               dd_mul(left.imag, right.imag)),
        dd_add(dd_mul(left.real, right.imag),
               dd_mul(left.imag, right.real)))


def cdd_div(left: ComplexDD, right: ComplexDD) -> ComplexDD:
    denominator = dd_add(dd_mul(right.real, right.real),
                         dd_mul(right.imag, right.imag))
    return ComplexDD(
        dd_div(dd_add(dd_mul(left.real, right.real),
                      dd_mul(left.imag, right.imag)), denominator),
        dd_div(dd_sub(dd_mul(left.imag, right.real),
                      dd_mul(left.real, right.imag)), denominator))


def cdd_div_corrected(left: complex, right: complex) -> ComplexDD:
    """One residual-corrected complex divide used by the native fast path."""
    inverse = 1.0 / (right.real * right.real + right.imag * right.imag)

    def divide_fast(numerator: complex) -> complex:
        return complex(
            (numerator.real * right.real
             + numerator.imag * right.imag) * inverse,
            (numerator.imag * right.real
             - numerator.real * right.imag) * inverse)

    first = divide_fast(left)
    product = ComplexDD(
        dd_sub(dd_mul(DD(first.real), DD(right.real)),
               dd_mul(DD(first.imag), DD(right.imag))),
        dd_add(dd_mul(DD(first.real), DD(right.imag)),
               dd_mul(DD(first.imag), DD(right.real))))
    residual = complex(
        dd_value(dd_sub(DD(left.real), product.real)),
        dd_value(dd_sub(DD(left.imag), product.imag)))
    return cdd_add(cdd(first), cdd(divide_fast(residual)))


def cdd_div_real(left: ComplexDD, right: DD) -> ComplexDD:
    return ComplexDD(dd_div(left.real, right), dd_div(left.imag, right))


def cdd_value(value: ComplexDD) -> complex:
    return complex(dd_value(value.real), dd_value(value.imag))


def build_image(partials: int, sections: int, alpha: float) -> Image:
    raw_nodes = [-ratio for ratio in ratios(sections)]
    raw_poles, raw_amps = source_bank(partials, alpha)
    nodes = [mp.mpf(value) for value in raw_nodes]
    source_poles = [mp.mpc(value) for value in raw_poles]
    source_amps = [mp.mpc(value) for value in raw_amps]
    zeros = [-node for node in nodes]

    def numerator(q):
        return product([q - zero for zero in zeros])

    def source_transform(q):
        return mp.fsum(amp / (q - pole)
                       for pole, amp in zip(source_poles, source_amps))

    def cluster_function(q):
        return numerator(q) * source_transform(q)

    wet_source_amps = [
        amp * product((pole - zero) / (pole + zero)
                      for zero in zeros)
        for pole, amp in zip(source_poles, source_amps)
    ]
    tail_residues = []
    for index, node in enumerate(nodes):
        denominator = product(node - other
                              for j, other in enumerate(nodes) if j != index)
        tail_residues.append(cluster_function(node) / denominator)

    divided = [cluster_function(node) for node in nodes]
    coefficients = [divided[0]]
    for order in range(1, sections):
        divided = [
            (divided[i + 1] - divided[i]) / (nodes[i + order] - nodes[i])
            for i in range(len(divided) - 1)
        ]
        coefficients.append(divided[0])

    scaled_weights = []
    for index, coefficient in enumerate(coefficients):
        transition_product = product(-nodes[j]
                                     for j in range(index, sections - 1))
        scaled_weights.append(coefficient / transition_product)
    return Image(nodes, source_poles, source_amps, wet_source_amps,
                 tail_residues, coefficients, scaled_weights)


def build_weights_dd(partials: int, sections: int, alpha: float
                     ) -> tuple[list[float], list[complex]]:
    """Build the phase-scaled Newton image in software double-double.

    This is the arithmetic intended for the float64 LLVM materializer.  Its
    final values cross to Metal only after stable whole-tail evaluation.
    """
    nodes = [DD(-ratio) for ratio in ratios(sections)]
    zeros = [dd_neg(node) for node in nodes]
    source_poles, source_amps = source_bank(partials, alpha)

    def cluster_function(q: DD) -> ComplexDD:
        numerator = DD(1.0)
        for zero in zeros:
            numerator = dd_mul(numerator, dd_sub(q, zero))
        source = cdd(0.0j)
        for pole, amp in zip(source_poles, source_amps):
            denominator = ComplexDD(
                dd_sub(q, DD(pole.real)), DD(-pole.imag))
            source = cdd_add(source, cdd_div(cdd(amp), denominator))
        return ComplexDD(dd_mul(source.real, numerator),
                         dd_mul(source.imag, numerator))

    divided = [cluster_function(node) for node in nodes]
    coefficients = [divided[0]]
    for order in range(1, sections):
        divided = [
            cdd_div_real(cdd_sub(divided[index + 1], divided[index]),
                         dd_sub(nodes[index + order], nodes[index]))
            for index in range(len(divided) - 1)
        ]
        coefficients.append(divided[0])
    weights = []
    for index, coefficient in enumerate(coefficients):
        transition_product = DD(1.0)
        for node in nodes[index:sections - 1]:
            transition_product = dd_mul(transition_product, dd_neg(node))
        weights.append(cdd_div_real(coefficient, transition_product))
    return ([dd_value(node) for node in nodes],
            [cdd_value(weight) for weight in weights])


def build_weights_corrected(partials: int, sections: int, alpha: float
                            ) -> tuple[list[float], list[complex]]:
    """Native-cost model: corrected source divides plus DD Newton differences."""
    nodes = [DD(-ratio) for ratio in ratios(sections)]
    zeros = [dd_neg(node) for node in nodes]
    source_poles, source_amps = source_bank(partials, alpha)

    def cluster_function(q: DD) -> ComplexDD:
        numerator = DD(1.0)
        for zero in zeros:
            numerator = dd_mul(numerator, dd_sub(q, zero))
        source = cdd(0.0j)
        for pole, amp in zip(source_poles, source_amps):
            denominator = complex(dd_value(q) - pole.real, -pole.imag)
            source = cdd_add(
                source, cdd_div_corrected(complex(amp), denominator))
        return ComplexDD(dd_mul(source.real, numerator),
                         dd_mul(source.imag, numerator))

    divided = [cluster_function(node) for node in nodes]
    coefficients = [divided[0]]
    for order in range(1, sections):
        divided = [
            cdd_div_real(cdd_sub(divided[index + 1], divided[index]),
                         dd_sub(nodes[index + order], nodes[index]))
            for index in range(len(divided) - 1)
        ]
        coefficients.append(divided[0])
    weights = []
    for index, coefficient in enumerate(coefficients):
        transition_product = DD(1.0)
        for node in nodes[index:sections - 1]:
            transition_product = dd_mul(transition_product, dd_neg(node))
        weights.append(cdd_div_real(coefficient, transition_product))
    return ([dd_value(node) for node in nodes],
            [cdd_value(weight) for weight in weights])


def build_weights_kahan(partials: int, sections: int, alpha: float
                        ) -> tuple[list[float], list[complex]]:
    """Cheap alternative: compensated double source sum, DD Newton table."""
    nodes = [DD(-ratio) for ratio in ratios(sections)]
    zeros = [dd_neg(node) for node in nodes]
    source_poles, source_amps = source_bank(partials, alpha)

    def cluster_function(q: DD) -> ComplexDD:
        numerator = DD(1.0)
        for zero in zeros:
            numerator = dd_mul(numerator, dd_sub(q, zero))
        total_real = total_imag = correction_real = correction_imag = 0.0
        for pole, amp in zip(source_poles, source_amps):
            quotient = complex(amp) / complex(dd_value(q) - pole.real,
                                               -pole.imag)
            add_real = quotient.real - correction_real
            next_real = total_real + add_real
            correction_real = (next_real - total_real) - add_real
            total_real = next_real
            add_imag = quotient.imag - correction_imag
            next_imag = total_imag + add_imag
            correction_imag = (next_imag - total_imag) - add_imag
            total_imag = next_imag
        source = cdd(complex(total_real, total_imag))
        return ComplexDD(dd_mul(source.real, numerator),
                         dd_mul(source.imag, numerator))

    divided = [cluster_function(node) for node in nodes]
    coefficients = [divided[0]]
    for order in range(1, sections):
        divided = [
            cdd_div_real(cdd_sub(divided[index + 1], divided[index]),
                         dd_sub(nodes[index + order], nodes[index]))
            for index in range(len(divided) - 1)
        ]
        coefficients.append(divided[0])
    weights = []
    for index, coefficient in enumerate(coefficients):
        transition_product = DD(1.0)
        for node in nodes[index:sections - 1]:
            transition_product = dd_mul(transition_product, dd_neg(node))
        weights.append(cdd_div_real(coefficient, transition_product))
    return ([dd_value(node) for node in nodes],
            [cdd_value(weight) for weight in weights])


def wet_source_amps_f64(partials: int, sections: int,
                        alpha: float) -> list[complex]:
    poles, amps = source_bank(partials, alpha)
    zeros = ratios(sections)
    return [
        amp * product((pole - zero) / (pole + zero) for zero in zeros)
        for pole, amp in zip(poles, amps)
    ]


def exp_dd_mp(nodes: list[mp.mpf], u: mp.mpf) -> mp.mpf:
    values = [mp.exp(node * u) for node in nodes]
    for order in range(1, len(nodes)):
        values = [(values[i + 1] - values[i]) /
                  (nodes[i + order] - nodes[i])
                  for i in range(len(values) - 1)]
    return values[0]


def complete_homogeneous(nodes: list[float], count: int) -> list[float]:
    values = [0.0] * count
    values[0] = 1.0
    for node in nodes:
        for degree in range(1, count):
            values[degree] += node * values[degree - 1]
    return values


def exp_dd_series(nodes: list[float], u: float, terms: int = 80) -> float:
    order = len(nodes) - 1
    homogeneous = complete_homogeneous(nodes, terms)
    total = 0.0
    power = u ** order
    factorial = math.factorial(order)
    for degree, coefficient in enumerate(homogeneous):
        if degree:
            power *= u
            factorial *= order + degree
        term = coefficient * power / factorial
        total += term
        if degree > 8 and abs(term) <= 2.0e-16 * max(1.0, abs(total)):
            break
    return total


def exp_dd_recursive(nodes: list[float], u: float) -> float:
    values = [math.exp(node * u) for node in nodes]
    for order in range(1, len(nodes)):
        values = [(values[i + 1] - values[i]) /
                  (nodes[i + order] - nodes[i])
                  for i in range(len(values) - 1)]
    return values[0]


def exp_dd_hybrid(nodes: list[float], u: float) -> float:
    if abs(u) * max(abs(node) for node in nodes) < 8.0:
        return exp_dd_series(nodes, u)
    return exp_dd_recursive(nodes, u)


def uniformized_suffix_carriers(nodes: list[float], u: float) -> tuple[list[float], int]:
    """Return bounded suffix carriers g[k] for all k in one positive fold.

    For rates a=-node, g[k] = prod(a[k:-1]) * exp[node[k],...,node[-1]].
    It is the k->last occupancy probability of a sequential phase-type chain.
    Uniformization evaluates the whole last column with non-negative arithmetic.
    """
    rates = [-node for node in nodes]
    rho = max(rates)
    mean = rho * u
    if mean > 700.0:
        return [0.0] * len(nodes), 0
    poisson = math.exp(-mean)
    power_column = [0.0] * len(nodes)
    power_column[-1] = 1.0
    result = [0.0] * len(nodes)
    limit = max(32, int(mean + 14.0 * math.sqrt(mean + 1.0) + 32.0))
    for step in range(limit):
        for index in range(len(nodes)):
            result[index] += poisson * power_column[index]
        old = power_column
        power_column = [0.0] * len(nodes)
        power_column[-1] = (1.0 - rates[-1] / rho) * old[-1]
        for index in range(len(nodes) - 2, -1, -1):
            power_column[index] = (
                (1.0 - rates[index] / rho) * old[index]
                + rates[index] / rho * old[index + 1])
        poisson *= mean / (step + 1.0)
    return result, limit


def cram16_suffix_carriers(nodes: list[float], u: float) -> list[float]:
    """Fixed-cost rational exp(u*T) last column for the phase-type chain."""
    rates = [-node for node in nodes]
    result = [0.0] * len(nodes)
    result[-1] = CRAM16_ALPHA0
    for alpha, theta in zip(CRAM16_ALPHA, CRAM16_THETA):
        solution = [0.0j] * len(nodes)
        for index in range(len(nodes) - 1, -1, -1):
            rhs = (1.0 if index == len(nodes) - 1 else 0.0)
            if index + 1 < len(nodes):
                rhs -= u * rates[index] * solution[index + 1]
            solution[index] = rhs / (u * nodes[index] - theta)
        for index, value in enumerate(solution):
            result[index] += 2.0 * (alpha * value).real
    return result


def cram48_ipf_suffix_carriers(nodes: list[float], u: float) -> list[float]:
    """Order-48 IPF CRAM evaluation of exp(u*T)'s last column."""
    rates = [-node for node in nodes]
    result = [0.0] * len(nodes)
    result[-1] = 1.0
    for alpha, theta in zip(CRAM48_ALPHA, CRAM48_THETA):
        solution = [0.0j] * len(nodes)
        for index in range(len(nodes) - 1, -1, -1):
            rhs = complex(result[index], 0.0)
            if index + 1 < len(nodes):
                rhs -= u * rates[index] * solution[index + 1]
            solution[index] = rhs / (u * nodes[index] - theta)
        for index, value in enumerate(solution):
            result[index] += 2.0 * (alpha * value).real
    return [CRAM48_ALPHA0 * value for value in result]


def complex_fsum(values) -> complex:
    values = list(values)
    return complex(math.fsum(value.real for value in values),
                   math.fsum(value.imag for value in values))


def render_mp(image: Image, u: float, mix: float = 0.5) -> mp.mpc:
    time = mp.mpf(u)
    source = mp.fsum(
        ((1.0 - mix) * original + mix * wet) * mp.exp(pole * time)
        for pole, original, wet in zip(
            image.source_poles, image.source_amps, image.wet_source_amps))
    tail = mp.fsum(residue * mp.exp(node * time)
                   for residue, node in zip(image.tail_residues, image.nodes))
    return source + mix * tail


def render_ordinary_f64(image: Image, u: float, mix: float = 0.5) -> complex:
    source = complex_fsum(
        ((1.0 - mix) * complex(original) + mix * complex(wet))
        * cmath.exp(complex(pole) * u)
        for pole, original, wet in zip(
            image.source_poles, image.source_amps, image.wet_source_amps))
    tail = complex_fsum(complex(residue) * math.exp(float(node) * u)
                        for residue, node in zip(image.tail_residues, image.nodes))
    return source + mix * tail


def render_newton_f64(image: Image, u: float, mix: float = 0.5) -> complex:
    source = complex_fsum(
        ((1.0 - mix) * complex(original) + mix * complex(wet))
        * cmath.exp(complex(pole) * u)
        for pole, original, wet in zip(
            image.source_poles, image.source_amps, image.wet_source_amps))
    nodes = [float(node) for node in image.nodes]
    contributions = []
    for index, coefficient in enumerate(image.newton_coefficients):
        basis = exp_dd_hybrid(nodes[index:], u)
        contributions.append(complex(coefficient) * basis)
    tail = complex_fsum(contributions)
    return source + mix * tail


def render_newton_oracle_carrier_f64(
        image: Image, u: float, mix: float = 0.5) -> complex:
    source = complex_fsum(
        ((1.0 - mix) * complex(original) + mix * complex(wet))
        * cmath.exp(complex(pole) * u)
        for pole, original, wet in zip(
            image.source_poles, image.source_amps, image.wet_source_amps))
    contributions = [
        complex(coefficient) * float(exp_dd_mp(image.nodes[index:], mp.mpf(u)))
        for index, coefficient in enumerate(image.newton_coefficients)
    ]
    return source + mix * complex_fsum(contributions)


def render_uniformized_f64(image: Image, u: float, mix: float = 0.5) -> tuple[complex, int]:
    source = complex_fsum(
        ((1.0 - mix) * complex(original) + mix * complex(wet))
        * cmath.exp(complex(pole) * u)
        for pole, original, wet in zip(
            image.source_poles, image.source_amps, image.wet_source_amps))
    carriers, iterations = uniformized_suffix_carriers(
        [float(node) for node in image.nodes], u)
    tail = complex_fsum(complex(weight) * carrier
                        for weight, carrier in zip(image.scaled_weights, carriers))
    return source + mix * tail, iterations


def render_cram16_f64(image: Image, u: float, mix: float = 0.5) -> complex:
    source = complex_fsum(
        ((1.0 - mix) * complex(original) + mix * complex(wet))
        * cmath.exp(complex(pole) * u)
        for pole, original, wet in zip(
            image.source_poles, image.source_amps, image.wet_source_amps))
    carriers = cram16_suffix_carriers([float(node) for node in image.nodes], u)
    tail = complex_fsum(complex(weight) * carrier
                        for weight, carrier in zip(image.scaled_weights, carriers))
    return source + mix * tail


def f32(value: float) -> float:
    """Round one operation result to the Metal scalar format."""
    try:
        return struct.unpack("<f", struct.pack("<f", float(value)))[0]
    except OverflowError:
        return math.copysign(math.inf, float(value))


def alpha_at(sample: float, center: float, sweep: float, rate: float) -> float:
    """Production phaser law, including its Q32 phase-increment truncation."""
    modulus = 4294967296
    increment = int(rate * modulus / SAMPLE_RATE)
    phase = ((increment * sample) % modulus) / modulus
    return 2.0 * math.pi * center * 2.0 ** (
        sweep * math.sin(2.0 * math.pi * phase))


def tail_at_mp(partials: int, sections: int, sample: float,
               center: float, sweep: float, rate: float) -> mp.mpc:
    alpha = alpha_at(sample, center, sweep, rate)
    image = build_image(partials, sections, alpha)
    u = mp.mpf(alpha) * mp.mpf(sample) / mp.mpf(SAMPLE_RATE)
    return mp.fsum(residue * mp.exp(node * u)
                   for residue, node in zip(image.tail_residues, image.nodes))


def tail_at_materializer_f64(partials: int, sections: int, sample: float,
                             center: float, sweep: float,
                             rate: float) -> complex:
    alpha = alpha_at(sample, center, sweep, rate)
    nodes, weights = build_weights_dd(partials, sections, alpha)
    u = alpha * sample / SAMPLE_RATE
    survival_bound = math.fsum(
        abs(weight) * math.exp(node * u)
        for weight, node in zip(weights, nodes))
    if survival_bound < MATERIALIZER_TAIL_CUTOFF:
        return 0.0j
    carriers, _ = uniformized_suffix_carriers(nodes, u)
    return complex_fsum(weight * carrier
                        for weight, carrier in zip(weights, carriers))


def tail_at_materializer_cram_f64(partials: int, sections: int, sample: float,
                                  center: float, sweep: float,
                                  rate: float) -> complex:
    """Fixed-cost support evaluator for the promotion-tolerance experiment."""
    alpha = alpha_at(sample, center, sweep, rate)
    nodes, weights = build_weights_dd(partials, sections, alpha)
    u = alpha * sample / SAMPLE_RATE
    survival_bound = math.fsum(
        abs(weight) * math.exp(node * u)
        for weight, node in zip(weights, nodes))
    if survival_bound < MATERIALIZER_TAIL_CUTOFF:
        return 0.0j
    carriers = cram16_suffix_carriers(nodes, u)
    return complex_fsum(weight * carrier
                        for weight, carrier in zip(weights, carriers))


def components_at_mp(partials: int, sections: int, sample: float,
                     center: float, sweep: float, rate: float
                     ) -> tuple[Image, mp.mpc, mp.mpc]:
    alpha = alpha_at(sample, center, sweep, rate)
    image = build_image(partials, sections, alpha)
    u = mp.mpf(alpha) * mp.mpf(sample) / mp.mpf(SAMPLE_RATE)
    source = mp.fsum(
        (mp.mpf("0.5") * original + mp.mpf("0.5") * wet)
        * mp.exp(pole * u)
        for pole, original, wet in zip(
            image.source_poles, image.source_amps, image.wet_source_amps))
    tail = mp.fsum(residue * mp.exp(node * u)
                   for residue, node in zip(image.tail_residues, image.nodes))
    return image, source, tail


def chebyshev_image(function, start: float, width: float,
                    count: int) -> list[complex]:
    """Gauss-node Chebyshev coefficients on one absolute sample interval."""
    values = []
    for index in range(count):
        angle = math.pi * (index + 0.5) / count
        coordinate = start + 0.5 * width * (1.0 + math.cos(angle))
        values.append(complex(function(coordinate)))
    coefficients = []
    for degree in range(count):
        scale = 1.0 / count if degree == 0 else 2.0 / count
        coefficients.append(scale * complex_fsum(
            value * math.cos(degree * math.pi * (index + 0.5) / count)
            for index, value in enumerate(values)))
    return coefficients


def chebyshev_vector_image(function, start: float, width: float,
                           count: int) -> list[list[complex]]:
    """One shared-node Chebyshev transform for a vector coefficient field."""
    values = []
    for index in range(count):
        angle = math.pi * (index + 0.5) / count
        coordinate = start + 0.5 * width * (1.0 + math.cos(angle))
        values.append([complex(value) for value in function(coordinate)])
    if not values:
        return []
    result: list[list[complex]] = []
    for row in range(len(values[0])):
        coefficients = []
        for degree in range(count):
            scale = 1.0 / count if degree == 0 else 2.0 / count
            coefficients.append(scale * complex_fsum(
                value[row] * math.cos(
                    degree * math.pi * (index + 0.5) / count)
                for index, value in enumerate(values)))
        result.append(coefficients)
    return result


def integer_chebyshev_nodes(width: int, count: int) -> list[int]:
    """Distinct integer offsets nearest an ordered Lobatto distribution."""
    targets = [
        0.5 * width * (1.0 - math.cos(math.pi * index / (count - 1)))
        for index in range(count)
    ]
    nodes: list[int] = []
    for index, target in enumerate(targets):
        lower = nodes[-1] + 1 if nodes else 0
        upper = width - (count - 1 - index)
        nodes.append(max(lower, min(upper, round(target))))
    return nodes


def solve_linear_f64(matrix: list[list[float]],
                     values: list[complex]) -> list[complex]:
    """Partial-pivoted solve used to model a precomputed inverse transform."""
    count = len(values)
    rows = [list(row) + [values[index]]
            for index, row in enumerate(matrix)]
    for column in range(count):
        pivot = max(range(column, count),
                    key=lambda row: abs(rows[row][column]))
        rows[column], rows[pivot] = rows[pivot], rows[column]
        divisor = rows[column][column]
        for index in range(column, count + 1):
            rows[column][index] /= divisor
        for row in range(column + 1, count):
            scale = rows[row][column]
            for index in range(column, count + 1):
                rows[row][index] -= scale * rows[column][index]
    result = [0.0j] * count
    for row in range(count - 1, -1, -1):
        result[row] = rows[row][count] - sum(
            rows[row][column] * result[column]
            for column in range(row + 1, count))
    return result


def integer_chebyshev_image(function, start: int, width: int,
                            count: int) -> list[complex]:
    """Chebyshev image from materializer-compatible integer coordinates."""
    offsets = integer_chebyshev_nodes(width, count)
    values = [complex(function(float(start + offset))) for offset in offsets]
    matrix = []
    for offset in offsets:
        coordinate = 2.0 * offset / width - 1.0
        row = [1.0]
        if count > 1:
            row.append(coordinate)
        for _ in range(2, count):
            row.append(2.0 * coordinate * row[-1] - row[-2])
        matrix.append(row)
    return solve_linear_f64(matrix, values)


def integer_chebyshev_vector_image(function, start: int, width: int,
                                   count: int) -> list[list[complex]]:
    offsets = integer_chebyshev_nodes(width, count)
    values = [[complex(value) for value in function(float(start + offset))]
              for offset in offsets]
    if not values:
        return []
    matrix = []
    for offset in offsets:
        coordinate = 2.0 * offset / width - 1.0
        row = [1.0]
        if count > 1:
            row.append(coordinate)
        for _ in range(2, count):
            row.append(2.0 * coordinate * row[-1] - row[-2])
        matrix.append(row)
    return [solve_linear_f64(matrix, [value[row] for value in values])
            for row in range(len(values[0]))]


def pack_complex_f32(values) -> bytes:
    """Serialize the proposed materializer/Metal complex-scalar boundary."""
    payload = bytearray()
    for value in values:
        payload.extend(struct.pack("<ff", f32(value.real), f32(value.imag)))
    return bytes(payload)


def final_segment_image_values(
        partials: int, sections: int, start: int, width: int,
        rate: float, center: float, sweep: float,
        weight_builder=build_weights_dd,
        ) -> tuple[list[list[complex]], list[complex]]:
    """Materialize one segment's source coefficients and whole-tail table.

    Only source wet-gain coefficients and bounded whole-tail samples cross this
    boundary.  Newton weights are a private float64/DD worker representation.
    """
    source_support = 10 if width == 128 else 8
    weight_support = 6 if width == 32 else 8
    source_images = integer_chebyshev_vector_image(
        lambda sample: wet_source_amps_f64(
            partials, sections, alpha_at(sample, center, sweep, rate)),
        start, width, min(source_support, width + 1))
    weight_images = integer_chebyshev_vector_image(
        lambda sample: weight_builder(
            partials, sections, alpha_at(sample, center, sweep, rate))[1],
        start, width, min(weight_support, width + 1))
    nodes = [-ratio for ratio in ratios(sections)]
    tail_values = []
    for offset in range(width):
        coordinate = 2.0 * offset / width - 1.0
        weights = [chebyshev_eval_f64(image, coordinate)
                   for image in weight_images]
        sample = start + offset
        alpha = alpha_at(sample, center, sweep, rate)
        u = alpha * sample / SAMPLE_RATE
        survival_bound = math.fsum(
            abs(weight) * math.exp(node * u)
            for weight, node in zip(weights, nodes))
        if survival_bound < MATERIALIZER_TAIL_CUTOFF:
            tail_values.append(0.0j)
        else:
            carriers, _ = uniformized_suffix_carriers(nodes, u)
            value = complex_fsum(
                weight * carrier
                for weight, carrier in zip(weights, carriers))
            tail_values.append(complex(f32(value.real), f32(value.imag)))
    return source_images, tail_values


def final_segment_image_bytes(
        partials: int, sections: int, start: int, width: int,
        rate: float, center: float, sweep: float) -> bytes:
    """Serialize one pure absolute-time image in its published precision."""
    source_images, tail_values = final_segment_image_values(
        partials, sections, start, width, rate, center, sweep)
    return pack_complex_f32(
        coefficient
        for source_image in source_images
        for coefficient in source_image) + pack_complex_f32(tail_values)


def run_determinism_experiment() -> None:
    """Check that scheduling order cannot affect published segment bytes."""
    cases = [
        (6, 6, 0, 128, 0.2, 700.0, 1.5),
        (32, 12, 0, 64, 0.2, 700.0, 1.5),
        (32, 18, 0, 32, 8.0, 4000.0, 3.0),
        (32, 18, 256, 128, 8.0, 40.0, 3.0),
        (32, 18, 1376, 32, 8.0, 700.0, 1.5),
        (32, 18, 4096, 32, 8.0, 700.0, 1.5),
    ]
    reference = {case: final_segment_image_bytes(*case) for case in cases}
    reversed_images = {
        case: final_segment_image_bytes(*case) for case in reversed(cases)
    }
    shuffled_order = [cases[index] for index in (3, 0, 5, 2, 1, 4)]
    shuffled_images = {
        case: final_segment_image_bytes(*case) for case in shuffled_order
    }
    repeated = {case: final_segment_image_bytes(*case) for case in cases}
    matches = all(
        reference[case] == reversed_images[case]
        == shuffled_images[case] == repeated[case]
        for case in cases)
    print(f"determinism cases={len(cases)}"
          f" orders=forward/reverse/shuffle/repeat byte-identical={matches}")
    for case in cases:
        digest = hashlib.sha256(reference[case]).hexdigest()[:16]
        print(f"  P={case[0]:2d} S={case[1]:2d} start={case[2]:4d}"
              f" I={case[3]:3d} bytes={len(reference[case]):5d}"
              f" sha256={digest}")


def barycentric_interpolate(nodes: list[int], values: list[complex],
                            coordinate: float) -> complex:
    """Evaluate the unique polynomial through the integer support image."""
    for node, value in zip(nodes, values):
        if coordinate == node:
            return value
    weighted = []
    for index, node in enumerate(nodes):
        denominator = 1.0
        for other_index, other in enumerate(nodes):
            if other_index != index:
                denominator *= node - other
        weighted.append(1.0 / denominator / (coordinate - node))
    return complex_fsum(weight * value
                        for weight, value in zip(weighted, values)) \
        / math.fsum(weighted)


def phaser_transfer(pole: complex, sections: int, alpha: float) -> complex:
    wet = product((pole - alpha * ratio) / (pole + alpha * ratio)
                  for ratio in ratios(sections))
    return 0.5 + 0.5 * wet


def run_notch_experiment() -> None:
    """Compare damped musical notch centers/depths over the legal control box.

    A small source damping avoids assigning an arbitrary dB depth to the
    mathematical zeros of an undamped all-pass/dry junction.  Exact and staged
    responses use the same dense logarithmic frequency grid.
    """
    scenarios = [
        (0, 0.02, 700.0, 1.5, 0.37),
        (256, 8.0, 40.0, 3.0, 0.61),
        (1376, 8.0, 700.0, 1.5, 0.43),
        (4096, 8.0, 4000.0, 3.0, 0.73),
    ]
    frequencies = [20.0 * (1000.0 ** (index / 8192.0))
                   for index in range(8193)]
    center_errors = []
    depth_errors = []
    compared = 0
    worst_center = None
    worst_depth = None
    for sections in (6, 12, 18):
        for width in (32, 64, 128):
            nodes = integer_chebyshev_nodes(width, 8)
            for start, rate, center, sweep, phase in scenarios:
                offset = phase * width
                sample = start + offset
                support_alphas = [alpha_at(start + node, center, sweep, rate)
                                  for node in nodes]
                exact_alpha = alpha_at(sample, center, sweep, rate)
                exact_magnitudes = []
                staged_magnitudes = []
                for frequency in frequencies:
                    pole = complex(-5.6, 2.0 * math.pi * frequency)
                    exact_magnitudes.append(abs(
                        phaser_transfer(pole, sections, exact_alpha)))
                    wet_support = [
                        2.0 * phaser_transfer(pole, sections, alpha) - 1.0
                        for alpha in support_alphas]
                    wet = barycentric_interpolate(nodes, wet_support, offset)
                    staged_magnitudes.append(abs(0.5 + 0.5 * wet))
                minima = [index for index in range(1, len(frequencies) - 1)
                          if exact_magnitudes[index] < exact_magnitudes[index - 1]
                          and exact_magnitudes[index]
                          <= exact_magnitudes[index + 1]]
                for exact_index in minima:
                    exact_frequency = frequencies[exact_index]
                    candidates = [index for index, frequency
                                  in enumerate(frequencies)
                                  if abs(frequency / exact_frequency - 1.0)
                                  <= 0.01]
                    staged_index = min(
                        candidates, key=lambda index: staged_magnitudes[index])
                    center_error = abs(
                        frequencies[staged_index] / exact_frequency - 1.0)
                    exact_db = 20.0 * math.log10(max(
                        exact_magnitudes[exact_index], 1.0e-12))
                    staged_db = 20.0 * math.log10(max(
                        staged_magnitudes[staged_index], 1.0e-12))
                    depth_error = abs(staged_db - exact_db)
                    center_errors.append(center_error)
                    depth_errors.append(depth_error)
                    compared += 1
                    identity = (sections, width, start, rate, center, sweep,
                                exact_frequency, frequencies[staged_index])
                    if worst_center is None or center_error > worst_center[0]:
                        worst_center = (center_error, identity)
                    if worst_depth is None or depth_error > worst_depth[0]:
                        worst_depth = (depth_error, exact_db, staged_db, identity)
    print(f"notch-audit comparisons={compared} damping=5.6"
          f" center-max={max(center_errors):.6e}"
          f" depth-max-db={max(depth_errors):.6e}"
          f" center-over-1pct={sum(value > 0.01 for value in center_errors)}"
          f" depth-over-1db={sum(value > 1.0 for value in depth_errors)}")
    print(f"  worst-center={worst_center}")
    print(f"  worst-depth={worst_depth}")


def final_segment_outputs(
        partials: int, sections: int, start: int, width: int,
        rate: float, center: float, sweep: float,
        weight_builder=build_weights_dd,
        ) -> tuple[list[complex], list[complex]]:
    """Render one published image and its multiprecision oracle samples."""
    source_images, tail_values = final_segment_image_values(
        partials, sections, start, width, rate, center, sweep,
        weight_builder)
    references = []
    candidates = []
    for offset in range(width):
        sample = start + offset
        image, source, tail = components_at_mp(
            partials, sections, float(sample), center, sweep, rate)
        reference = complex(source + mp.mpf("0.5") * tail)
        alpha = alpha_at(sample, center, sweep, rate)
        u = alpha * sample / SAMPLE_RATE
        coordinate = 2.0 * offset / width - 1.0
        wet = [chebyshev_eval_f32(row, coordinate)
               for row in source_images]
        staged_source = complex_fsum(
            (0.5 * complex(original) + 0.5 * wet_value)
            * cmath.exp(complex(pole) * u)
            for pole, original, wet_value in zip(
                image.source_poles, image.source_amps, wet))
        candidate = staged_source + 0.5 * tail_values[offset]
        if sample <= 0:
            reference = 0.0j
            candidate = 0.0j
        references.append(reference)
        candidates.append(candidate)
    return references, candidates


def final_segment_error_samples(
        partials: int, sections: int, start: int, width: int,
        rate: float, center: float, sweep: float,
        weight_builder=build_weights_dd,
        ) -> tuple[list[float], list[float], float]:
    """Compare the exact published image against the multiprecision oracle."""
    references, candidates = final_segment_outputs(
        partials, sections, start, width, rate, center, sweep,
        weight_builder)
    peak = max((abs(value) for value in references), default=0.0)
    absolute = [abs(candidate - reference)
                for candidate, reference in zip(candidates, references)]
    denominator = max(peak, 1.0e-30)
    return [error / denominator for error in absolute], absolute, peak


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[round(quantile * (len(ordered) - 1))]


def run_randomized_distribution_experiment(
        count: int = 96, weight_builder=build_weights_dd) -> None:
    """Seeded continuous-control audit of error distribution and maxima."""
    rng = random.Random(0xA85E_2026)
    topologies = [
        (6, 6), (6, 12), (6, 18),
        (32, 6), (32, 12), (32, 18),
    ]
    widths = [32, 64, 128]
    sample_relative: list[float] = []
    sample_absolute: list[float] = []
    segment_maxima: list[float] = []
    worst = None
    for index in range(count):
        partials, sections = topologies[index % len(topologies)]
        width = rng.choice(widths)
        center = 2.0 ** rng.uniform(math.log2(40.0), math.log2(4000.0))
        sweep = rng.uniform(0.0, 3.0)
        rate = 2.0 ** rng.uniform(math.log2(0.02), math.log2(8.0))
        if index % 4 == 0:
            start = 0
        else:
            latest = min(round(SAMPLE_RATE / rate), 16384)
            start = rng.randrange(latest + 1)
        relative, absolute, peak = final_segment_error_samples(
            partials, sections, start, width, rate, center, sweep,
            weight_builder)
        maximum = max(relative, default=0.0)
        sample_relative.extend(relative)
        sample_absolute.extend(absolute)
        segment_maxima.append(maximum)
        if worst is None or maximum > worst[0]:
            worst = (maximum, max(absolute, default=0.0), peak,
                     partials, sections, start, width, rate, center, sweep)
    print(f"random-distribution segments={count} samples={len(sample_relative)}")
    print("  sample-relative " + " ".join(
        f"p{round(100 * quantile):02d}={percentile(sample_relative, quantile):.6e}"
        for quantile in (0.5, 0.9, 0.95, 0.99))
        + f" max={max(sample_relative):.6e}"
        + f" >1e-5={sum(value > 1.0e-5 for value in sample_relative)}"
        + f" >1e-4={sum(value > 1.0e-4 for value in sample_relative)}")
    print("  segment-maxima  " + " ".join(
        f"p{round(100 * quantile):02d}={percentile(segment_maxima, quantile):.6e}"
        for quantile in (0.5, 0.9, 0.95, 0.99))
        + f" max={max(segment_maxima):.6e}"
        + f" >1e-5={sum(value > 1.0e-5 for value in segment_maxima)}"
        + f" >1e-4={sum(value > 1.0e-4 for value in segment_maxima)}")
    assert worst is not None
    print("  worst"
          f" relative={worst[0]:.6e} absolute={worst[1]:.6e}"
          f" peak={worst[2]:.6e} P={worst[3]} S={worst[4]}"
          f" start={worst[5]} I={worst[6]} rate={worst[7]:.6g}"
          f" center={worst[8]:.6g} sweep={worst[9]:.6g}")


def run_stratified_matrix_experiment(
        starts: tuple[int, ...] = (0, 256, 1376, 4096),
        weight_builder=build_weights_corrected,
        output_gain: float = 3.7) -> None:
    """Audit every admitted discrete shape/control row at active starts.

    This is the multiprecision counterpart to the 162-row harness matrix.  It
    deliberately samples independently materialized intervals rather than
    rendering the ill-conditioned incumbent 32-partial bank as an oracle.  The
    fixed starts keep the impulse response active for slow LFO cases instead
    of spending most of the audit on numerically tiny, already-decayed tails.
    """
    topologies = [
        (6, 6), (6, 12), (6, 18),
        (32, 6), (32, 12), (32, 18),
    ]
    widths = (32, 64, 128)
    rates = (0.02, 0.2, 8.0)
    controls = ((700.0, 1.5), (40.0, 0.0), (4000.0, 3.0))
    sample_relative: list[float] = []
    sample_absolute: list[float] = []
    segment_relative: list[float] = []
    segment_absolute: list[float] = []
    topology_maxima = {topology: 0.0 for topology in topologies}
    worst_relative = None
    worst_absolute = None
    rows = 0
    for partials, sections in topologies:
        for width in widths:
            for rate in rates:
                for center, sweep in controls:
                    for start in starts:
                        references, candidates = final_segment_outputs(
                            partials, sections, start, width, rate,
                            center, sweep, weight_builder)
                        peak = max((abs(output_gain * value.real)
                                    for value in references), default=0.0)
                        absolute = [
                            abs(output_gain * (candidate - reference).real)
                            for candidate, reference
                            in zip(candidates, references)]
                        denominator = max(peak, 1.0e-30)
                        relative = [error / denominator for error in absolute]
                        relative_max = max(relative, default=0.0)
                        absolute_max = max(absolute, default=0.0)
                        identity = (partials, sections, start, width, rate,
                                    center, sweep, peak)
                        sample_relative.extend(relative)
                        sample_absolute.extend(absolute)
                        segment_relative.append(relative_max)
                        segment_absolute.append(absolute_max)
                        topology_maxima[(partials, sections)] = max(
                            topology_maxima[(partials, sections)], absolute_max)
                        if (worst_relative is None
                                or relative_max > worst_relative[0]):
                            worst_relative = (relative_max, absolute_max,
                                              identity)
                        if (worst_absolute is None
                                or absolute_max > worst_absolute[0]):
                            worst_absolute = (absolute_max, relative_max,
                                              identity)
                        rows += 1
                        if rows % 72 == 0:
                            print(f"  stratified progress={rows}", flush=True)
    assert worst_relative is not None and worst_absolute is not None
    print(f"stratified-matrix rows={rows} samples={len(sample_relative)}"
          f" starts={','.join(str(start) for start in starts)}"
          f" output-gain={output_gain:g}")
    print("  sample-relative " + " ".join(
        f"p{round(100 * quantile):02d}="
        f"{percentile(sample_relative, quantile):.6e}"
        for quantile in (0.5, 0.9, 0.95, 0.99))
        + f" max={max(sample_relative):.6e}"
        + f" >1e-5={sum(value > 1.0e-5 for value in sample_relative)}"
        + f" >1e-4={sum(value > 1.0e-4 for value in sample_relative)}")
    print("  sample-absolute " + " ".join(
        f"p{round(100 * quantile):02d}="
        f"{percentile(sample_absolute, quantile):.6e}"
        for quantile in (0.5, 0.9, 0.95, 0.99))
        + f" max={max(sample_absolute):.6e}"
        + f" >1e-5={sum(value > 1.0e-5 for value in sample_absolute)}"
        + f" >1e-4={sum(value > 1.0e-4 for value in sample_absolute)}")
    print("  segment-relative " + " ".join(
        f"p{round(100 * quantile):02d}="
        f"{percentile(segment_relative, quantile):.6e}"
        for quantile in (0.5, 0.9, 0.95, 0.99))
        + f" max={max(segment_relative):.6e}"
        + f" >1e-4={sum(value > 1.0e-4 for value in segment_relative)}")
    print("  segment-absolute " + " ".join(
        f"p{round(100 * quantile):02d}="
        f"{percentile(segment_absolute, quantile):.6e}"
        for quantile in (0.5, 0.9, 0.95, 0.99))
        + f" max={max(segment_absolute):.6e}"
        + f" >1e-4={sum(value > 1.0e-4 for value in segment_absolute)}")
    print("  topology-absolute-max " + " ".join(
        f"P{partials}S{sections}={topology_maxima[(partials, sections)]:.6e}"
        for partials, sections in topologies))
    print(f"  worst-relative={worst_relative}")
    print(f"  worst-absolute={worst_absolute}")


def run_boundary_transient_experiment(
        weight_builder=build_weights_dd) -> None:
    """Audit first-difference error at independently materialized boundaries."""
    cases = [
        (6, 6, 0, 128, 4, 8.0, 4000.0, 3.0),
        (6, 18, 0, 128, 4, 8.0, 4000.0, 3.0),
        (32, 18, 0, 128, 4, 8.0, 4000.0, 3.0),
        (32, 18, 0, 32, 16, 8.0, 4000.0, 3.0),
        (32, 18, 256, 128, 4, 8.0, 40.0, 3.0),
        (32, 18, 1376, 32, 8, 8.0, 700.0, 1.5),
        (32, 12, 4096, 64, 4, 0.02, 700.0, 1.5),
    ]
    all_boundary_relative = []
    all_boundary_envelope = []
    print(f"boundary-transients cases={len(cases)}")
    for (partials, sections, start, width, segment_count,
         rate, center, sweep) in cases:
        references = []
        candidates = []
        for segment in range(segment_count):
            segment_reference, segment_candidate = final_segment_outputs(
                partials, sections, start + segment * width, width,
                rate, center, sweep, weight_builder)
            references.extend(segment_reference)
            candidates.extend(segment_candidate)
        peak = max((abs(value) for value in references), default=0.0)
        denominator = max(peak, 1.0e-30)
        reference_differences = [
            abs(references[index] - references[index - 1])
            for index in range(1, len(references))
        ]
        difference_errors = [
            abs((candidates[index] - candidates[index - 1])
                - (references[index] - references[index - 1]))
            for index in range(1, len(references))
        ]
        boundary_indices = [segment * width
                            for segment in range(1, segment_count)]
        boundary_errors = [difference_errors[index - 1]
                           for index in boundary_indices]
        envelope_ratios = []
        for index, error in zip(boundary_indices, boundary_errors):
            radius = min(221, width)
            local = reference_differences[
                max(0, index - radius):min(
                    len(reference_differences), index + radius)]
            envelope = max(local, default=0.0)
            envelope_ratios.append(error / max(envelope, 1.0e-30))
        boundary_relative = [error / denominator
                             for error in boundary_errors]
        all_boundary_relative.extend(boundary_relative)
        all_boundary_envelope.extend(envelope_ratios)
        print(f"  P={partials:2d} S={sections:2d} start={start:4d}"
              f" I={width:3d} segments={segment_count:2d}"
              f" boundary-diff/peak={max(boundary_relative):.6e}"
              f" all-diff/peak={max(difference_errors) / denominator:.6e}"
              f" boundary/local-envelope={max(envelope_ratios):.6e}")
    print(f"  aggregate boundaries={len(all_boundary_relative)}"
          f" max-diff/peak={max(all_boundary_relative):.6e}"
          f" max/local-envelope={max(all_boundary_envelope):.6e}"
          f" >1e-4={sum(value > 1.0e-4 for value in all_boundary_relative)}")


def shared_weight_cram_integer_image(
        partials: int, sections: int, start: int, width: int,
        center: float, sweep: float, rate: float,
        weight_count: int = 4, tail_count: int = 24) -> list[complex]:
    """Integer tail image with a shared, smooth internal Newton-weight field."""
    weight_images = integer_chebyshev_vector_image(
        lambda sample: build_weights_dd(
            partials, sections, alpha_at(sample, center, sweep, rate))[1],
        start, width, weight_count)
    nodes = [-ratio for ratio in ratios(sections)]
    tail_offsets = integer_chebyshev_nodes(width, tail_count)
    tail_values = []
    for offset in tail_offsets:
        coordinate = 2.0 * offset / width - 1.0
        weights = [chebyshev_eval_f64(image, coordinate)
                   for image in weight_images]
        sample = start + offset
        alpha = alpha_at(sample, center, sweep, rate)
        u = alpha * sample / SAMPLE_RATE
        survival_bound = math.fsum(
            abs(weight) * math.exp(node * u)
            for weight, node in zip(weights, nodes))
        if survival_bound < MATERIALIZER_TAIL_CUTOFF:
            tail_values.append(0.0j)
        else:
            carriers = cram16_suffix_carriers(nodes, u)
            tail_values.append(complex_fsum(
                weight * carrier
                for weight, carrier in zip(weights, carriers)))
    matrix = []
    for offset in tail_offsets:
        coordinate = 2.0 * offset / width - 1.0
        row = [1.0]
        if tail_count > 1:
            row.append(coordinate)
        for _ in range(2, tail_count):
            row.append(2.0 * coordinate * row[-1] - row[-2])
        matrix.append(row)
    return solve_linear_f64(matrix, tail_values)


def chebyshev_eval_f64(coefficients: list[complex], coordinate: float) -> complex:
    """Clenshaw evaluation for sum c[k] T_k(coordinate)."""
    following = 0.0j
    after_following = 0.0j
    for coefficient in reversed(coefficients[1:]):
        current = 2.0 * coordinate * following - after_following + coefficient
        after_following = following
        following = current
    return coordinate * following - after_following + coefficients[0]


def chebyshev_eval_f32(coefficients: list[complex], coordinate: float) -> complex:
    x = f32(coordinate)
    following = 0.0j
    after_following = 0.0j
    for raw in reversed(coefficients[1:]):
        coefficient = complex(f32(raw.real), f32(raw.imag))
        current = complex(
            f32(f32(f32(2.0 * x) * following.real) -
                after_following.real + coefficient.real),
            f32(f32(f32(2.0 * x) * following.imag) -
                after_following.imag + coefficient.imag))
        after_following = following
        following = current
    first = complex(f32(coefficients[0].real), f32(coefficients[0].imag))
    return complex(
        f32(f32(x * following.real) - after_following.real + first.real),
        f32(f32(x * following.imag) - after_following.imag + first.imag))


def run_local_tail_case(partials: int, sections: int, interval: int,
                        rate: float, center: float, sweep: float) -> None:
    """Measure a bounded time-local whole-tail representation at the attack."""
    exact = {
        sample: complex(tail_at_mp(
            partials, sections, float(sample), center, sweep, rate))
        for sample in range(interval + 1)
    }
    peak = max((abs(value) for value in exact.values()), default=1.0)
    print(f"local-tail P={partials:2d} S={sections:2d} I={interval:3d}"
          f" rate={rate:g} center={center:g} sweep={sweep:g}"
          f" peak={peak:.6e}")
    function = lambda sample: tail_at_mp(
        partials, sections, sample, center, sweep, rate)
    for count in (4, 8, 12, 16, 24, 32):
        coefficients = chebyshev_image(function, 0.0, float(interval), count)
        error64 = 0.0
        error32 = 0.0
        worst64 = 0
        worst32 = 0
        for sample, reference in exact.items():
            coordinate = 2.0 * sample / interval - 1.0
            candidate64 = chebyshev_eval_f64(coefficients, coordinate)
            candidate32 = chebyshev_eval_f32(coefficients, coordinate)
            if abs(candidate64 - reference) > error64:
                error64 = abs(candidate64 - reference)
                worst64 = sample
            if abs(candidate32 - reference) > error32:
                error32 = abs(candidate32 - reference)
                worst32 = sample
        coefficient_peak = max(abs(value) for value in coefficients)
        print(f"  cheb-{count:2d} coeff-peak={coefficient_peak:.6e}"
              f" f64={error64 / peak:.6e}@{worst64}"
              f" f32={error32 / peak:.6e}@{worst32}")


def lerp_complex_f32(left: complex, right: complex, phase: float) -> complex:
    t = f32(phase)
    one_minus = f32(1.0 - t)
    return complex(
        f32(f32(one_minus * f32(left.real)) + f32(t * f32(right.real))),
        f32(f32(one_minus * f32(left.imag)) + f32(t * f32(right.imag))))


def run_local_total_case(partials: int, sections: int, interval: int,
                         rate: float, center: float, sweep: float,
                         start: int, count: int) -> None:
    """Combine staged source gains with the local whole-tail image."""
    start_image, _, start_tail = components_at_mp(
        partials, sections, float(start), center, sweep, rate)
    end_image, _, end_tail = components_at_mp(
        partials, sections, float(start + interval), center, sweep, rate)
    function = lambda sample: tail_at_mp(
        partials, sections, sample, center, sweep, rate)
    coefficients = chebyshev_image(
        function, float(start), float(interval), count)
    materializer_coefficients = chebyshev_image(
        lambda sample: tail_at_materializer_f64(
            partials, sections, sample, center, sweep, rate),
        float(start), float(interval), count)
    cram_coefficients = chebyshev_image(
        lambda sample: tail_at_materializer_cram_f64(
            partials, sections, sample, center, sweep, rate),
        float(start), float(interval), count)
    wet_function = lambda sample: wet_source_amps_f64(
        partials, sections, alpha_at(sample, center, sweep, rate))
    wet_images = {
        source_count: chebyshev_vector_image(
            wet_function, float(start), float(interval), source_count)
        for source_count in (2, 4, 6, 8)
    }
    integer_tail_count = min(24, interval + 1)
    integer_cram_coefficients = integer_chebyshev_image(
        lambda sample: tail_at_materializer_cram_f64(
            partials, sections, sample, center, sweep, rate),
        start, interval, integer_tail_count)
    integer_wet_images = integer_chebyshev_vector_image(
        wet_function, start, interval, min(6, interval + 1))
    shared_integer_coefficients = shared_weight_cram_integer_image(
        partials, sections, start, interval, center, sweep, rate,
        min(4, interval + 1), integer_tail_count)
    final_source_images = integer_chebyshev_vector_image(
        wet_function, start, interval, min(8, interval + 1))
    final_weight_images = integer_chebyshev_vector_image(
        lambda sample: build_weights_dd(
            partials, sections, alpha_at(sample, center, sweep, rate))[1],
        start, interval, min(6, interval + 1))
    final_nodes = [-ratio for ratio in ratios(sections)]
    peak = 0.0
    linear_tail_error = 0.0
    staged64_error = 0.0
    staged32_error = 0.0
    worst32 = start
    source_cheb_errors = {source_count: 0.0 for source_count in wet_images}
    source_cheb_worst = {source_count: start for source_count in wet_images}
    realizable_error = 0.0
    materializer_worst = start
    cram_materializer_error = 0.0
    cram_materializer_worst = start
    integer_materializer_error = 0.0
    integer_materializer_worst = start
    shared_materializer_error = 0.0
    shared_materializer_worst = start
    final_materializer_error = 0.0
    final_materializer_worst = start
    for sample in range(start, start + interval + 1):
        image, source, tail = components_at_mp(
            partials, sections, float(sample), center, sweep, rate)
        reference = complex(source + mp.mpf("0.5") * tail)
        if sample <= 0:
            reference = 0.0j
        peak = max(peak, abs(reference))
        phase = (sample - start) / interval
        coordinate = 2.0 * phase - 1.0
        tail_linear = (1.0 - phase) * complex(start_tail) + \
            phase * complex(end_tail)
        linear_tail_error = max(
            linear_tail_error, abs(0.5 * (tail_linear - complex(tail))))
        u = alpha_at(sample, center, sweep, rate) * sample / SAMPLE_RATE
        source64 = complex_fsum(
            (0.5 * complex(original) + 0.5 * (
                (1.0 - phase) * complex(start_wet) +
                phase * complex(end_wet))) * cmath.exp(complex(pole) * u)
            for pole, original, start_wet, end_wet in zip(
                image.source_poles, image.source_amps,
                start_image.wet_source_amps, end_image.wet_source_amps))
        source32 = complex_fsum(
            (0.5 * complex(original) + 0.5 * lerp_complex_f32(
                complex(start_wet), complex(end_wet), phase))
            * cmath.exp(complex(pole) * u)
            for pole, original, start_wet, end_wet in zip(
                image.source_poles, image.source_amps,
                start_image.wet_source_amps, end_image.wet_source_amps))
        candidate64 = source64 + 0.5 * chebyshev_eval_f64(
            coefficients, coordinate)
        candidate32 = source32 + 0.5 * chebyshev_eval_f32(
            coefficients, coordinate)
        if sample <= 0:
            candidate64 = 0.0j
            candidate32 = 0.0j
        staged64_error = max(staged64_error, abs(candidate64 - reference))
        error32 = abs(candidate32 - reference)
        if error32 > staged32_error:
            staged32_error = error32
            worst32 = sample
        for source_count, row_images in wet_images.items():
            wet = [chebyshev_eval_f32(row, coordinate) for row in row_images]
            source_cheb = complex_fsum(
                (0.5 * complex(original) + 0.5 * wet_value)
                * cmath.exp(complex(pole) * u)
                for pole, original, wet_value in zip(
                    image.source_poles, image.source_amps, wet))
            candidate = source_cheb + 0.5 * chebyshev_eval_f32(
                coefficients, coordinate)
            if sample <= 0:
                candidate = 0.0j
            error = abs(candidate - reference)
            if error > source_cheb_errors[source_count]:
                source_cheb_errors[source_count] = error
                source_cheb_worst[source_count] = sample
            if source_count == 6:
                materialized = source_cheb + 0.5 * chebyshev_eval_f32(
                    materializer_coefficients, coordinate)
                if sample <= 0:
                    materialized = 0.0j
                error = abs(materialized - reference)
                if error > realizable_error:
                    realizable_error = error
                    materializer_worst = sample
                cram_materialized = source_cheb + 0.5 * chebyshev_eval_f32(
                    cram_coefficients, coordinate)
                if sample <= 0:
                    cram_materialized = 0.0j
                error = abs(cram_materialized - reference)
                if error > cram_materializer_error:
                    cram_materializer_error = error
                    cram_materializer_worst = sample
        integer_wet = [chebyshev_eval_f32(row, coordinate)
                       for row in integer_wet_images]
        integer_source = complex_fsum(
            (0.5 * complex(original) + 0.5 * wet_value)
            * cmath.exp(complex(pole) * u)
            for pole, original, wet_value in zip(
                image.source_poles, image.source_amps, integer_wet))
        integer_candidate = integer_source + 0.5 * chebyshev_eval_f32(
            integer_cram_coefficients, coordinate)
        if sample <= 0:
            integer_candidate = 0.0j
        error = abs(integer_candidate - reference)
        if error > integer_materializer_error:
            integer_materializer_error = error
            integer_materializer_worst = sample
        shared_candidate = integer_source + 0.5 * chebyshev_eval_f32(
            shared_integer_coefficients, coordinate)
        if sample <= 0:
            shared_candidate = 0.0j
        error = abs(shared_candidate - reference)
        if error > shared_materializer_error:
            shared_materializer_error = error
            shared_materializer_worst = sample
        final_wet = [chebyshev_eval_f32(row, coordinate)
                     for row in final_source_images]
        final_source = complex_fsum(
            (0.5 * complex(original) + 0.5 * wet_value)
            * cmath.exp(complex(pole) * u)
            for pole, original, wet_value in zip(
                image.source_poles, image.source_amps, final_wet))
        final_weights = [chebyshev_eval_f64(row, coordinate)
                         for row in final_weight_images]
        survival_bound = math.fsum(
            abs(weight) * math.exp(node * u)
            for weight, node in zip(final_weights, final_nodes))
        if survival_bound < MATERIALIZER_TAIL_CUTOFF:
            final_tail = 0.0j
        else:
            carriers, _ = uniformized_suffix_carriers(final_nodes, u)
            final_tail = complex_fsum(
                weight * carrier
                for weight, carrier in zip(final_weights, carriers))
        final_tail = complex(f32(final_tail.real), f32(final_tail.imag))
        final_candidate = final_source + 0.5 * final_tail
        if sample <= 0:
            final_candidate = 0.0j
        error = abs(final_candidate - reference)
        if error > final_materializer_error:
            final_materializer_error = error
            final_materializer_worst = sample
    denominator = max(peak, 1.0e-30)
    print(f"local-total P={partials:2d} S={sections:2d} I={interval:3d}"
          f" start={start:6d} cheb={count:2d} peak={peak:.6e}"
          f" tail-linear/peak={linear_tail_error / denominator:.6e}"
          f" staged-f64/peak={staged64_error / denominator:.6e}"
          f" staged-f32/peak={staged32_error / denominator:.6e}@{worst32}")
    print("  source-cheb " + " ".join(
        f"{source_count}:{source_cheb_errors[source_count] / denominator:.6e}"
        f"@{source_cheb_worst[source_count]}"
        for source_count in wet_images))
    print(f"  realizable DD/uniform + source-6:"
          f" {realizable_error / denominator:.6e}@{materializer_worst}")
    print(f"  fixed DD/CRAM-16 + source-6:"
          f" {cram_materializer_error / denominator:.6e}"
          f"@{cram_materializer_worst}")
    print(f"  integer tail-{integer_tail_count}/source-6:"
          f" {integer_materializer_error / denominator:.6e}"
          f"@{integer_materializer_worst}")
    print(f"  shared weight-4/tail-{integer_tail_count}/source-6:"
          f" {shared_materializer_error / denominator:.6e}"
          f"@{shared_materializer_worst}")
    print(f"  final integer table/uniform weight-6/source-8:"
          f" {final_materializer_error / denominator:.6e}"
          f"@{final_materializer_worst}")


def samples() -> list[float]:
    dense = [index * 0.025 for index in range(401)]
    tail = [10.0 * 1.035 ** index for index in range(90)]
    return dense + tail


def magnitude(values) -> float:
    return max((abs(complex(value)) for value in values), default=0.0)


def run_case(partials: int, sections: int) -> None:
    # alpha factors out of the tail topology.  The source pole normalization
    # still uses the production center at the zero-LFO coordinate.
    alpha = 2.0 * math.pi * 700.0
    image = build_image(partials, sections, alpha)
    dd_nodes, dd_weights = build_weights_dd(partials, sections, alpha)
    peak = 0.0
    ordinary_error = 0.0
    newton_error = 0.0
    oracle_carrier_error = 0.0
    uniformized_error = 0.0
    cram16_error = 0.0
    materializer_error = 0.0
    ordinary_nonfinite = 0
    newton_nonfinite = 0
    worst_newton_u = 0.0
    max_uniformization_iterations = 0
    for u in samples():
        reference = complex(render_mp(image, u))
        ordinary = render_ordinary_f64(image, u)
        newton = render_newton_f64(image, u)
        oracle_carrier = render_newton_oracle_carrier_f64(image, u)
        uniformized, iterations = render_uniformized_f64(image, u)
        cram16 = render_cram16_f64(image, u)
        dd_carriers, _ = uniformized_suffix_carriers(dd_nodes, u)
        dd_tail = complex_fsum(weight * carrier
                               for weight, carrier in zip(
                                   dd_weights, dd_carriers))
        exact_tail = complex(mp.fsum(
            residue * mp.exp(node * mp.mpf(u))
            for residue, node in zip(image.tail_residues, image.nodes)))
        max_uniformization_iterations = max(max_uniformization_iterations, iterations)
        peak = max(peak, abs(reference))
        if math.isfinite(ordinary.real) and math.isfinite(ordinary.imag):
            ordinary_error = max(ordinary_error, abs(ordinary - reference))
        else:
            ordinary_nonfinite += 1
        if math.isfinite(newton.real) and math.isfinite(newton.imag):
            error = abs(newton - reference)
            if error > newton_error:
                newton_error = error
                worst_newton_u = u
        else:
            newton_nonfinite += 1
        oracle_carrier_error = max(
            oracle_carrier_error, abs(oracle_carrier - reference))
        uniformized_error = max(uniformized_error, abs(uniformized - reference))
        cram16_error = max(cram16_error, abs(cram16 - reference))
        materializer_error = max(materializer_error, abs(dd_tail - exact_tail))
    print(f"P={partials:2d} S={sections:2d}")
    print(f"  max ordinary tail residue  {magnitude(image.tail_residues):.6e}")
    print(f"  max Newton coefficient     {magnitude(image.newton_coefficients):.6e}")
    print(f"  max phase-scaled weight    {magnitude(image.scaled_weights):.6e}")
    print(f"  ordinary max/peak error    {ordinary_error / peak:.6e}"
          f"  nonfinite={ordinary_nonfinite}")
    print(f"  Newton max/peak error      {newton_error / peak:.6e}"
          f"  nonfinite={newton_nonfinite}  at u={worst_newton_u:.6g}")
    print(f"  Newton exact-carrier floor {oracle_carrier_error / peak:.6e}")
    print(f"  phase-type uniformization {uniformized_error / peak:.6e}"
          f"  max-iterations={max_uniformization_iterations}")
    print(f"  phase-type CRAM-16        {cram16_error / peak:.6e}"
          f"  triangular-solves={len(CRAM16_ALPHA)}")
    print(f"  DD-build + uniformization {materializer_error / peak:.6e}")


def main() -> None:
    mp.mp.dps = MP_DPS
    for partials, sections in ((6, 6), (32, 12), (32, 18)):
        run_case(partials, sections)
    for case in (
        (6, 6, 128, 0.2, 700.0, 1.5),
        (6, 6, 64, 8.0, 40.0, 3.0),
        (6, 6, 32, 8.0, 4000.0, 3.0),
        (32, 12, 64, 0.2, 700.0, 1.5),
        (32, 18, 32, 8.0, 700.0, 1.5),
    ):
        run_local_tail_case(*case)
    for case in (
        (6, 6, 128, 0.2, 700.0, 1.5, 0, 32),
        (6, 6, 64, 8.0, 40.0, 3.0, 0, 24),
        (6, 6, 32, 8.0, 4000.0, 3.0, 0, 32),
        (32, 12, 64, 0.2, 700.0, 1.5, 0, 24),
        (32, 18, 32, 8.0, 700.0, 1.5, 0, 24),
        (6, 6, 32, 8.0, 4000.0, 3.0, 1376, 16),
        (32, 18, 32, 8.0, 700.0, 1.5, 1376, 16),
        (32, 18, 32, 8.0, 700.0, 1.5, 4096, 16),
    ):
        run_local_total_case(*case)
    run_determinism_experiment()


if __name__ == "__main__":
    main()
