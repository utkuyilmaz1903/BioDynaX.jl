using Pkg
Pkg.develop(PackageSpec(path = dirname(@__DIR__)))
Pkg.resolve()
Pkg.instantiate()
