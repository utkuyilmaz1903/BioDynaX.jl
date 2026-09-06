#!/usr/bin/env julia
# Download the single-cell p53 traces of the p53–Mdm2 case study.
#
# Source: the p53 dataset inside the Mendeley Data deposit "CODEX, a neural
# network approach to explore signaling dynamics landscapes" (Jacques,
# Gagliardi, Dobrzynski; version 2, 2021-02-01, doi:10.17632/4vnndy59fp.2),
# the supporting data of Jacques et al. (2021), Mol. Syst. Biol. 17:e10026.
# The deposit's archive `data_forCNN.zip` contains `p53_DoseCellLine.zip`
# with `dataset.csv` (3825 single-cell p53-YFP traces, 12 cell lines, 5 doses
# of ionizing radiation, 96 points at 15 min from 0 to 1425 min), `classes.csv`
# (cell line and dose per class) and `id_set.csv`. The deposit describes the
# p53 data as "kindly provided by Galit Lahav and Jacob Ornstein-Stewart"; the
# traces are those of Stewart-Ornstein and Lahav (2017), "Dynamics of p53 in
# response to DNA damage vary across cell lines and are shaped by efficiency
# of DNA repair and activity of the kinase ATM", Science Signaling 10,
# eaah6671, doi:10.1126/scisignal.aah6671 (p53-YFP reporter, 24 h imaging
# after 1, 2, 4, 6, or 8 Gy). Only p53 is observed; there is no Mdm2 reporter.
#
# Licence: the deposit is CC BY 4.0. Its licence text adds that "further
# permission may be required for any content within the dataset that is
# identified as belonging to a third party", and the p53 data are identified
# as provided by a third party, so the traces are not redistributed with
# BioDynaX: this script downloads the archive from the deposit, checks the
# SHA-256 of the archive and of the inner p53 file, and unpacks the three CSV
# files into examples/p53_mdm2/data/ (not committed). Cite both papers above
# and the deposit when using the data.
#
# Run:  julia --project=. examples/p53_mdm2/download_data.jl
# Needs network access (about 200 MB) and the `unzip` command. Not run by the
# test suite or CI.

using Downloads
using SHA

const P53_ARCHIVE_URL = "https://data.mendeley.com/public-files/datasets/4vnndy59fp/files/ff44cff2-8004-46a2-805a-e78afab41c3d/file_downloaded"
const P53_ARCHIVE_SHA256 = "34562085ca7a3eaf0693d6a752e0da521f05ad6bc338e136f8208a209f2e29a7"
const P53_INNER_ZIP = "p53_DoseCellLine.zip"
const P53_INNER_SHA256 = "8335f946f691e67f07f8fa658e3744460083585cd22b7fcadcccfcfe9e81f3df"
const P53_DATA_DIR = joinpath(@__DIR__, "data")
const P53_ARCHIVE = joinpath(P53_DATA_DIR, "data_forCNN.zip")
const P53_UNPACKED = joinpath(P53_DATA_DIR, "p53_DoseCellLine")

sha256_hex(path) = bytes2hex(open(sha256, path))

"""
    download_p53_data(; force=false) -> String

Download the deposit archive to `examples/p53_mdm2/data/` (reused when its
checksum already matches), verify it, extract and verify the inner p53 zip,
unpack `dataset.csv`, `classes.csv`, and `id_set.csv`, and return the
directory holding them.
"""
function download_p53_data(; force::Bool = false)
    mkpath(P53_DATA_DIR)
    inner = joinpath(P53_DATA_DIR, P53_INNER_ZIP)
    if force || !isfile(inner) || sha256_hex(inner) != P53_INNER_SHA256
        if force || !isfile(P53_ARCHIVE) || sha256_hex(P53_ARCHIVE) != P53_ARCHIVE_SHA256
            println("downloading ", P53_ARCHIVE_URL, " (about 200 MB)")
            Downloads.download(P53_ARCHIVE_URL, P53_ARCHIVE)
        end
        digest = sha256_hex(P53_ARCHIVE)
        digest == P53_ARCHIVE_SHA256 || error(
            "checksum mismatch for $(P53_ARCHIVE): expected $(P53_ARCHIVE_SHA256), got $(digest)")
        println("verified archive SHA-256 ", digest)
        run(`unzip -q -o $(P53_ARCHIVE) $(P53_INNER_ZIP) -d $(P53_DATA_DIR)`)
    end
    digest = sha256_hex(inner)
    digest == P53_INNER_SHA256 || error(
        "checksum mismatch for $(inner): expected $(P53_INNER_SHA256), got $(digest)")
    println("verified ", P53_INNER_ZIP, " SHA-256 ", digest)
    if force || !isfile(joinpath(P53_UNPACKED, "dataset.csv"))
        rm(P53_UNPACKED; force = true, recursive = true)
        mkpath(P53_UNPACKED)
        run(`unzip -q -o $(inner) -d $(P53_UNPACKED)`)
    end
    println("unpacked into ", P53_UNPACKED)
    println("licence: CC BY 4.0 deposit with third-party content; the traces stay in ",
        "this directory and are not committed")
    return P53_UNPACKED
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    download_p53_data()
end
