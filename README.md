---
# Use with pandoc ... --pdf-engine=xelatex
mainfont: Times Roman
geometry: left=2cm,right=2cm,top=2cm,bottom=2cm
fontsize: 12pt
---

# Charge-conserving denoiser
One way to impose exact charge conservation in a convolution is to ensure that all the kernel elements
add to 1.  Let $K_I$ be the kernel elements where $I$ is a 2d index tuple. If $q_I$ is the original 
charge density in pixel $I$ after convolution we have
$$
q'_J = \sum_I q_I K_{J - I}.
$$
Charge conservation is ensured because
$$
\sum_J q'_J = \sum_J \sum_I q_I K_{J-I} = \sum_I \left(\sum_J K_{J-I}\right) q_I = \sum_I q_I.
$$
