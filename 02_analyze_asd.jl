using Distributed
@everywhere using Printf, DelimitedFiles, Random, Statistics, ProgressMeter, Base.Threads
@everywhere include("../MDToolbox.jl/src/MDToolbox.jl")
@everywhere using .MDToolbox
@everywhere using BSON: @save, @load
@everywhere include("01_getposterior_parallel_use_local_afm.jl")


function analyze()
    asd_file_name = "201711020044";
    asd = readasd("data/t1r/$(asd_file_name).asd");
    qs = readdlm("data/quaternion/QUATERNION_LIST_576_Orient")
    models = readpdb("data/t1r/cluster.pdb");
    for iatom = 1:models.natom
        models.atomname[iatom] = models.resname[iatom]
    end
    MDToolbox.decenter!(models)
    nq = size(qs, 1)
    pred_radius = 15
    nframe = size(asd.frames, 1)
    params = [AfmizeConfig(10.0 * (pi / 180),
        r, 
        MDToolbox.Point2D(-250, -200), 
        MDToolbox.Point2D(250, 200), 
        MDToolbox.Point2D(6.25, 6.25), 
        MDToolbox.defaultParameters())
        for r in [pred_radius]]

    afms = []
    for i = 1:size(asd.frames, 1)
        push!(afms, asd.frames[i].data)
    end

    r = getposterior_parallel_use_local_afm(models, afms, qs, params)
    @save "data/02_result/$(asd_file_name)/radius_$(pred_radius).bson" r params
end

analyze()