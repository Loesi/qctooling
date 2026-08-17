/*
 * Native molden parser exposing a C ABI for use through Python ctypes.
 *
 * Entry points:
 *   MoldenResult *molden_parse(const char *path);
 *   void molden_result_free(MoldenResult *r);
 *
 * The parser reads the file line-by-line. Any line whose first non-whitespace
 * character is '[' starts a section; a switch dispatches to the matching
 * section parser, passing the line reader so parsing continues exactly where
 * it left off. Returned buffers are allocated with malloc and owned by the
 * caller of molden_parse; release everything with molden_result_free.
 *
 * All floats are read with strtod, which is forced to the "C" locale so a
 * process locale with a comma decimal separator cannot corrupt parsing.
 */

#include <ctype.h>
#include <errno.h>
#include <locale.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXLINE 65536

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

/* ------------------------------------------------------------------ */
/* Growable buffer                                                     */
/* ------------------------------------------------------------------ */

typedef struct {
    void *data;
    size_t n;
    size_t cap;
    size_t esz;
} Buffer;

static void buf_init(Buffer *b, size_t esz) {
    b->esz = esz;
    b->data = NULL;
    b->n = 0;
    b->cap = 0;
}

static int set_oom(MoldenResult *res) {
    if (res->status == 0) {
        res->status = 1;
        res->error = strdup("out of memory");
    }
    return -1;
}

static int buf_push(Buffer *b, const void *item, MoldenResult *res) {
    if (b->n == b->cap) {
        size_t ncap = b->cap ? b->cap * 2 : 16;
        void *nd = realloc(b->data, ncap * b->esz);
        if (!nd)
            return set_oom(res);
        b->data = nd;
        b->cap = ncap;
    }
    memcpy((char *)b->data + b->n * b->esz, item, b->esz);
    b->n++;
    return 0;
}

static void buf_free(Buffer *b) {
    free(b->data);
    b->data = NULL;
    b->n = b->cap = 0;
}

/* Free a buffer of char* (each string + the array). */
static void buf_free_strings(Buffer *b) {
    for (size_t i = 0; i < b->n; i++)
        free(((char **)b->data)[i]);
    buf_free(b);
}

/* ------------------------------------------------------------------ */
/* Line reader with one-line pushback                                  */
/* ------------------------------------------------------------------ */

typedef struct {
    FILE *f;
    char buf[MAXLINE];
    int has_pending;
} Reader;

/* Return next line, or NULL at EOF. Overlong lines are truncated but the
 * remainder is drained so section boundaries are never missed. */
static char *reader_next(Reader *r) {
    if (r->has_pending) {
        r->has_pending = 0;
        return r->buf;
    }
    if (!fgets(r->buf, (int)sizeof(r->buf), r->f))
        return NULL;
    size_t len = strlen(r->buf);
    if (len > 0 && r->buf[len - 1] != '\n') {
        int c;
        while ((c = fgetc(r->f)) != '\n' && c != EOF)
            ;
    }
    return r->buf;
}

static void reader_push(Reader *r) { r->has_pending = 1; }

/* ------------------------------------------------------------------ */
/* Small string helpers                                                */
/* ------------------------------------------------------------------ */

static int starts_ci(const char *s, const char *prefix) {
    while (*prefix) {
        if (tolower((unsigned char)*s) != tolower((unsigned char)*prefix))
            return 0;
        s++;
        prefix++;
    }
    return 1;
}

static char *trim(char *s) {
    while (*s && isspace((unsigned char)*s))
        s++;
    char *e = s + strlen(s);
    while (e > s && isspace((unsigned char)e[-1]))
        *--e = '\0';
    return s;
}

/* Split on whitespace in place, up to `max` tokens. Returns token count. */
static int split(char *s, char **tok, int max) {
    int n = 0;
    char *p = s;
    while (*p) {
        while (*p && isspace((unsigned char)*p))
            p++;
        if (!*p)
            break;
        if (n >= max)
            break;
        tok[n++] = p;
        while (*p && !isspace((unsigned char)*p))
            p++;
        if (*p)
            *p++ = '\0';
    }
    return n;
}

/* s/p/d/f/g/... -> l, -1 if unknown */
static int l_index(const char *s) {
    switch (tolower((unsigned char)s[0])) {
    case 's': return 0;
    case 'p': return 1;
    case 'd': return 2;
    case 'f': return 3;
    case 'g': return 4;
    case 'h': return 5;
    case 'i': return 6;
    default:  return -1;
    }
}

static int is_section_header(const char *line) {
    while (*line && isspace((unsigned char)*line))
        line++;
    return *line == '[';
}

static int is_blank(const char *line) {
    while (*line && isspace((unsigned char)*line))
        line++;
    return *line == '\0';
}

/* ------------------------------------------------------------------ */
/* Error reporting                                                     */
/* ------------------------------------------------------------------ */

static void fail(MoldenResult *res, const char *fmt, ...) {
    if (res->status)
        return;
    res->status = 1;
    char tmp[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(tmp, sizeof(tmp), fmt, ap);
    va_end(ap);
    res->error = strdup(tmp);
}

/* ------------------------------------------------------------------ */
/* Data collected during parsing (buffers)                             */
/* ------------------------------------------------------------------ */

typedef struct {
    Buffer elements, coords;
    Buffer shell_atom, shell_n, shell_l, shell_nprim, shell_off;
    Buffer alpha, coeff;
    Buffer tags;
    Buffer C, occ, ene, spin, irrep;
    int gto_done; /* tags are only collected between [GTO] and [MO] */
    int mo_done;
} Data;

static void data_init(Data *d) {
    buf_init(&d->elements, sizeof(char *));
    buf_init(&d->coords, sizeof(double));
    buf_init(&d->shell_atom, sizeof(int32_t));
    buf_init(&d->shell_n, sizeof(int32_t));
    buf_init(&d->shell_l, sizeof(int32_t));
    buf_init(&d->shell_nprim, sizeof(int32_t));
    buf_init(&d->shell_off, sizeof(int32_t));
    buf_init(&d->alpha, sizeof(double));
    buf_init(&d->coeff, sizeof(double));
    buf_init(&d->tags, sizeof(char *));
    buf_init(&d->C, sizeof(double));
    buf_init(&d->occ, sizeof(double));
    buf_init(&d->ene, sizeof(double));
    buf_init(&d->spin, sizeof(int32_t));
    buf_init(&d->irrep, sizeof(char *));
    d->gto_done = 0;
    d->mo_done = 0;
}

static void data_free(Data *d) {
    buf_free_strings(&d->elements);
    buf_free(&d->coords);
    buf_free(&d->shell_atom);
    buf_free(&d->shell_n);
    buf_free(&d->shell_l);
    buf_free(&d->shell_nprim);
    buf_free(&d->shell_off);
    buf_free(&d->alpha);
    buf_free(&d->coeff);
    buf_free_strings(&d->tags);
    buf_free(&d->C);
    buf_free(&d->occ);
    buf_free(&d->ene);
    buf_free(&d->spin);
    buf_free_strings(&d->irrep);
}

/* Move buffer ownership into the result. */
#define TAKE(res, res_field, buf, T)         \
    (res)->res_field = (T)(buf)->data;       \
    (buf)->data = NULL;                      \
    (buf)->n = (buf)->cap = 0;

/* ------------------------------------------------------------------ */
/* Section parsers                                                     */
/* ------------------------------------------------------------------ */

static void parse_atoms(Reader *r, Data *d, MoldenResult *res) {
    char *line;
    while ((line = reader_next(r))) {
        if (is_section_header(line)) {
            reader_push(r);
            return;
        }
        char *tok[16];
        int nt = split(line, tok, 16);
        if (nt < 6)
            continue;
        char *el = strdup(tok[0]);
        if (!el) {
            set_oom(res);
            return;
        }
        buf_push(&d->elements, &el, res);
        for (int i = 0; i < 3; i++) {
            double c = strtod(tok[3 + i], NULL);
            buf_push(&d->coords, &c, res);
        }
        if (res->status)
            return;
    }
}

static void parse_gto(Reader *r, Data *d, MoldenResult *res) {
    char *line;
    while ((line = reader_next(r))) {
        if (is_section_header(line)) {
            reader_push(r);
            return;
        }
        char *tok[16];
        int nt = split(line, tok, 16);
        if (nt < 1)
            continue;
        long atom_idx = strtol(tok[0], NULL, 10) - 1; /* 1-based -> 0 */
        /* Per-atom shell counters starting at l, matching the reference
         * Python parser (np.arange(5) then +1 per shell of that l). */
        int counter[7] = {0, 1, 2, 3, 4, 5, 6};

        while ((line = reader_next(r))) {
            if (is_section_header(line)) {
                reader_push(r);
                return;
            }
            if (is_blank(line))
                break; /* blank line ends this atom's shells */
            char *st[16];
            int sn = split(line, st, 16);
            if (sn < 2) {
                fail(res, "malformed shell line: %s", line);
                return;
            }
            int l = l_index(st[0]);
            if (l < 0) {
                fail(res, "unknown shell type '%s'", st[0]);
                return;
            }
            long nprim = strtol(st[1], NULL, 10);
            if (nprim < 0 || nprim > 100000) {
                fail(res, "implausible primitive count %ld", nprim);
                return;
            }
            int off = (int)d->alpha.n;
            for (long k = 0; k < nprim; k++) {
                line = reader_next(r);
                if (!line || is_section_header(line)) {
                    fail(res, "unexpected end of primitive block");
                    return;
                }
                char *pt[16];
                int pn = split(line, pt, 16);
                if (pn < 2) {
                    fail(res, "malformed primitive line: %s", line);
                    return;
                }
                double a = strtod(pt[0], NULL);
                double c = strtod(pt[1], NULL);
                buf_push(&d->alpha, &a, res);
                buf_push(&d->coeff, &c, res);
                if (res->status)
                    return;
            }
            int n = counter[l] + 1;
            counter[l]++;
            int32_t v;
            v = (int32_t)atom_idx;
            buf_push(&d->shell_atom, &v, res);
            v = n;
            buf_push(&d->shell_n, &v, res);
            v = l;
            buf_push(&d->shell_l, &v, res);
            v = (int32_t)nprim;
            buf_push(&d->shell_nprim, &v, res);
            v = off;
            buf_push(&d->shell_off, &v, res);
            if (res->status)
                return;
        }
    }
}

static void parse_mo(Reader *r, Data *d, MoldenResult *res) {
    int n_ao = 0;
    for (size_t i = 0; i < d->shell_l.n; i++)
        n_ao += 2 * ((int32_t *)d->shell_l.data)[i] + 1;
    if (n_ao <= 0) {
        fail(res, "no basis read before [MO] section");
        return;
    }
    int count[2] = {0, 0};
    char *line;

    while ((line = reader_next(r))) {
        if (is_section_header(line)) {
            reader_push(r);
            break;
        }
        if (is_blank(line))
            continue;

        /* Read up to 4 header lines (Sym/Ene/Spin/Occup). */
        char sym[256] = {0}, spinlabel[32] = {0};
        double ene = 0.0, occ = 0.0;
        int got = 0;
        while (got < 4 && line && !is_section_header(line)) {
            char *h = trim(line);
            if (starts_ci(h, "Sym=")) {
                snprintf(sym, sizeof(sym), "%s", trim(h + 4));
                got++;
            } else if (starts_ci(h, "Ene=")) {
                ene = strtod(h + 4, NULL);
                got++;
            } else if (starts_ci(h, "Spin=")) {
                snprintf(spinlabel, sizeof(spinlabel), "%s", trim(h + 5));
                got++;
            } else if (starts_ci(h, "Occup=")) {
                occ = strtod(h + 6, NULL);
                got++;
            } else {
                break; /* first coefficient line */
            }
            line = reader_next(r);
        }
        if (!line)
            break;

        int row = (strncasecmp(spinlabel, "beta", 4) == 0) ? 1 : 0;
        int j = count[row];
        if (j >= n_ao) {
            fail(res, "too many MOs for spin block %d", row);
            return;
        }

        buf_push(&d->occ, &occ, res);
        buf_push(&d->ene, &ene, res);
        int32_t sp = (row == 0);
        buf_push(&d->spin, &sp, res);
        char *irp = strdup(sym);
        if (!irp) {
            set_oom(res);
            return;
        }
        buf_push(&d->irrep, &irp, res);
        if (res->status)
            return;

        int l;
        for (l = 0; l < n_ao; l++) {
            char *ct[16];
            int cn = split(line, ct, 16);
            if (cn < 1) {
                fail(res, "malformed coefficient line: %s", line);
                return;
            }
            double v = strtod(ct[cn - 1], NULL);
            buf_push(&d->C, &v, res);
            if (res->status)
                return;
            if (l + 1 < n_ao) {
                line = reader_next(r);
                if (!line)
                    break;
                if (is_section_header(line)) {
                    reader_push(r);
                    line = NULL;
                    break;
                }
            }
        }
        if (l == n_ao)
            count[row]++; /* only count a fully-read MO */
        if (line == NULL)
            break;
    }

    if (count[0] != n_ao) {
        fail(res, "expected %d alpha MOs, found %d", n_ao, count[0]);
        return;
    }
    int n_spin = (count[1] > 0) ? 2 : 1;
    if (n_spin == 2 && count[1] != n_ao) {
        fail(res, "expected %d beta MOs, found %d", n_ao, count[1]);
        return;
    }
    res->n_ao = n_ao;
    res->n_spin = n_spin;
}

/* ------------------------------------------------------------------ */
/* Entry points                                                        */
/* ------------------------------------------------------------------ */

static void finalize(MoldenResult *res, Data *d) {
    res->n_atom = (int32_t)d->elements.n;
    res->n_shell = (int32_t)d->shell_atom.n;
    res->n_tags = (int32_t)d->tags.n;
    TAKE(res, elements, &d->elements, char **);
    TAKE(res, coords, &d->coords, double *);
    TAKE(res, shell_atom, &d->shell_atom, int32_t *);
    TAKE(res, shell_n, &d->shell_n, int32_t *);
    TAKE(res, shell_l, &d->shell_l, int32_t *);
    TAKE(res, shell_nprim, &d->shell_nprim, int32_t *);
    TAKE(res, shell_off, &d->shell_off, int32_t *);
    TAKE(res, alpha, &d->alpha, double *);
    TAKE(res, coeff, &d->coeff, double *);
    TAKE(res, tags, &d->tags, char **);
    TAKE(res, C, &d->C, double *);
    TAKE(res, occ, &d->occ, double *);
    TAKE(res, ene, &d->ene, double *);
    TAKE(res, spin, &d->spin, int32_t *);
    TAKE(res, irrep, &d->irrep, char **);
}

MoldenResult *molden_parse(const char *path) {
    MoldenResult *res = calloc(1, sizeof(MoldenResult));
    if (!res)
        return NULL;

    static int locale_done = 0;
    if (!locale_done) {
        setlocale(LC_NUMERIC, "C");
        locale_done = 1;
    }

    FILE *f = fopen(path, "rb");
    if (!f) {
        fail(res, "cannot open '%s': %s", path, strerror(errno));
        return res;
    }

    Data d;
    data_init(&d);
    Reader r;
    r.f = f;
    r.has_pending = 0;

    char *line;
    while ((line = reader_next(&r))) {
        if (!is_section_header(line))
            continue;
        if (starts_ci(line, "[Atoms]")) {
            parse_atoms(&r, &d, res);
        } else if (starts_ci(line, "[GTO]")) {
            parse_gto(&r, &d, res);
            if (res->status == 0)
                d.gto_done = 1;
        } else if (starts_ci(line, "[MO]")) {
            parse_mo(&r, &d, res);
            d.mo_done = 1;
        } else if (starts_ci(line, "[Pseudo]")) {
            /* Pseudo potentials are currently ignored; skip the section. */
            while ((line = reader_next(&r))) {
                if (is_section_header(line)) {
                    reader_push(&r);
                    break;
                }
            }
        } else {
            /* Unknown section header, e.g. "[5D]", "[7F]", "[Title]".
             * Only headers between [GTO] and [MO] are recorded as tags. */
            if (d.gto_done && !d.mo_done) {
                char *t = strdup(trim(line));
                if (t) {
                    buf_push(&d.tags, &t, res);
                    if (res->status) {
                        free(t);
                    }
                } else {
                    set_oom(res);
                }
            }
        }
        if (res->status)
            break;
    }

    fclose(f);

    if (res->status == 0) {
        finalize(res, &d);
    }
    data_free(&d);
    return res;
}

void molden_result_free(MoldenResult *r) {
    if (!r)
        return;
    free(r->error);
    if (r->elements) {
        for (int32_t i = 0; i < r->n_atom; i++)
            free(r->elements[i]);
        free(r->elements);
    }
    free(r->coords);
    free(r->shell_atom);
    free(r->shell_n);
    free(r->shell_l);
    free(r->shell_nprim);
    free(r->shell_off);
    free(r->alpha);
    free(r->coeff);
    if (r->tags) {
        for (int32_t i = 0; i < r->n_tags; i++)
            free(r->tags[i]);
        free(r->tags);
    }
    free(r->C);
    free(r->occ);
    free(r->ene);
    free(r->spin);
    if (r->irrep) {
        for (int32_t i = 0; i < r->n_spin * r->n_ao; i++)
            free(r->irrep[i]);
        free(r->irrep);
    }
    free(r);
}
