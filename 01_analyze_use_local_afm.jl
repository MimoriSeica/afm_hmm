using Distributed
@everywhere using Printf, DelimitedFiles, Random, Statistics, ProgressMeter, Base.Threads
@everywhere include("../MDToolbox.jl/src/MDToolbox.jl")
@everywhere using .MDToolbox
@everywhere using BSON: @save, @load
@everywhere include("01_getposterior_parallel_use_local_afm.jl")

function analyze()
    qs = readdlm("data/quaternion/QUATERNION_LIST_576_Orient")
    models = readpdb("data/t1r/cluster.pdb");
    for iatom = 1:models.natom
        models.atomname[iatom] = models.resname[iatom]
    end
    MDToolbox.decenter!(models)
    nq = size(qs, 1)
    test_radius = 20
    pred_radius = 25
    sigma_noise = 3.0
    nframe = 100
    run(`mkdir -p data/01_result/test_radius_$(test_radius)_pred_radius_$(pred_radius)`)

    params = [AfmizeConfig(10.0 * (pi / 180),
        r, 
        MDToolbox.Point2D(-250, -200), 
        MDToolbox.Point2D(250, 200), 
        MDToolbox.Point2D(6.25, 6.25), 
        MDToolbox.defaultParameters())
        for r in [pred_radius]]

    for iq in 1:nq
        @show iq
        @load "data/01_test_case/radius_$(test_radius)/iq_$(iq)_noise_$(sigma_noise)_nframe_$(nframe).bson" models afms qs

        # all models with parallel map
        r = getposterior_parallel_use_local_afm(models, afms, qs, params)

        @save "data/01_result/test_radius_$(test_radius)_pred_radius_$(pred_radius)/iq_$(iq)_nframe_$(nframe).bson" params r
    end
end

@time analyze()