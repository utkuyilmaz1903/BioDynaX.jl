#!/usr/bin/env julia
# Download the laccase/ABTS progress-curve dataset of the enzyme case study.
#
# Source: EnzymeML document `Scenario4/ABTS_Measurement_Ngubane.omex` in the
# GitHub repository EnzymeML/Lauterbach_2022, the supporting repository of
# Lauterbach et al. (2023), "EnzymeML: seamless data flow and modeling of
# enzymatic data", Nature Methods 20, 400-402, doi:10.1038/s41592-022-01763-1.
# Document creator: Sandile Ngubane (metadata.rdf, 2021-10-15). The file is
# fetched at a pinned commit and checked against its SHA-256 before use.
#
# Licence: the repository carries no licence file and the document no licence
# statement, so the data are not redistributed with BioDynaX; this script
# downloads them from the original source into examples/laccase_abts/data/,
# which is not committed. The article itself is open access (CC BY 4.0); that
# licence covers the article, not necessarily the repository.
#
# Run:  julia --project=. examples/laccase_abts/download_data.jl
# Needs network access and the `unzip` command. Not run by the test suite or CI.

using Downloads
using SHA

const ABTS_COMMIT = "348b742f3c5f7e4e0d0a679b22ccd6b4d9bfdbe3"
const ABTS_URL = string("https://raw.githubusercontent.com/EnzymeML/Lauterbach_2022/",
    ABTS_COMMIT, "/Scenario4/ABTS_Measurement_Ngubane.omex")
const ABTS_SHA256 = "d026cb2038c27e1bb7ceb7f4d60686b9a6feec6e92d1817aa9e6f9c09ff54975"
const ABTS_DATA_DIR = joinpath(@__DIR__, "data")
const ABTS_OMEX = joinpath(ABTS_DATA_DIR, "ABTS_Measurement_Ngubane.omex")
const ABTS_UNPACKED = joinpath(ABTS_DATA_DIR, "omex")

sha256_hex(path) = bytes2hex(open(sha256, path))

"""
    download_abts_data(; force=false) -> String

Download the EnzymeML document to `examples/laccase_abts/data/`, verify its
SHA-256, unpack it with `unzip`, and return the unpacked directory. An
existing verified download is reused unless `force` is true.
"""
function download_abts_data(; force::Bool = false)
    mkpath(ABTS_DATA_DIR)
    if force || !isfile(ABTS_OMEX) || sha256_hex(ABTS_OMEX) != ABTS_SHA256
        println("downloading ", ABTS_URL)
        Downloads.download(ABTS_URL, ABTS_OMEX)
    end
    digest = sha256_hex(ABTS_OMEX)
    digest == ABTS_SHA256 || error(
        "checksum mismatch for $(ABTS_OMEX): expected $(ABTS_SHA256), got $(digest)")
    println("verified SHA-256 ", digest)
    if force || !isfile(joinpath(ABTS_UNPACKED, "experiment.xml"))
        rm(ABTS_UNPACKED; force = true, recursive = true)
        mkpath(ABTS_UNPACKED)
        run(`unzip -q -o $(ABTS_OMEX) -d $(ABTS_UNPACKED)`)
    end
    println("unpacked into ", ABTS_UNPACKED)
    println("licence: none stated by the source; the data stay in this directory ",
        "and are not committed")
    return ABTS_UNPACKED
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    download_abts_data()
end
