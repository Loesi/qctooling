#ifndef MOLDEN_HELPER_H
#define MOLDEN_HELPER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MoldenResult {
    int32_t status;        /* 0 = ok */
    char *error;           /* malloc'd, NULL when ok */
    int32_t n_atom;
    char **elements;       /* [n_atom] element symbols */
    double *coords;        /* [n_atom * 3] */
    int32_t n_shell;
    int32_t *shell_atom;   /* [n_shell] atom index (0-based) */
    int32_t *shell_n;      /* [n_shell] principal shell counter */
    int32_t *shell_l;      /* [n_shell] azimuthal quantum number */
    int32_t *shell_nprim;  /* [n_shell] number of primitives */
    int32_t *shell_off;    /* [n_shell] offset into alpha/coeff */
    double *alpha;         /* [total primitives], raw exponents */
    double *coeff;         /* [total primitives], raw unnormalized coeffs */
    int32_t n_tags;        /* section tags, e.g. "[5D]" */
    char **tags;
    int32_t n_ao;          /* number of AOs (spherical) */
    int32_t n_spin;        /* 1 (restricted) or 2 (unrestricted) */
    double *C;             /* [n_spin * n_ao * n_ao] */
    double *occ;           /* [n_spin * n_ao] */
    double *ene;           /* [n_spin * n_ao] */
    int32_t *spin;         /* [n_spin * n_ao] 1 = alpha, 0 = beta */
    char **irrep;          /* [n_spin * n_ao] symmetry labels */
} MoldenResult;

/* Parse a molden file. Returns a heap-allocated result (never NULL on
 * success). On failure status != 0 and error is set. Always release with
 * molden_result_free. */
MoldenResult *molden_parse(const char *path);

/* Free a result returned by molden_parse. Safe to call with NULL. */
void molden_result_free(MoldenResult *r);

#ifdef __cplusplus
}
#endif

#endif /* MOLDEN_HELPER_H */
