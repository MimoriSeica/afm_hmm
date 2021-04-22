using Printf, ProgressMeter, DelimitedFiles
include("../MDToolbox.jl/src/MDToolbox.jl")
using .MDToolbox
using BSON: @save, @load

qs = readdlm("data/quaternion/QUATERNION_LIST_576_Orient")
nq = size(qs, 1)
models = readpdb("data/t1r/cluster.pdb")
for iatom = 1:models.natom
    models.atomname[iatom] = models.resname[iatom]
end
MDToolbox.decenter!(models)
nmodel = size(models, 1)
radius = 15

params = [AfmizeConfig(10.0 * (pi / 180),
    r, 
    MDToolbox.Point2D(-250, -200), 
    MDToolbox.Point2D(250, 200), 
    MDToolbox.Point2D(6.25, 6.25), 
    MDToolbox.defaultParameters())
    for r in [radius]]
nparam = length(params)

Threads.@threads for imodel in 1:nmodel
    model = models[imodel, :]
    for iq in 1:nq
        q = qs[iq, :]
        model_rotated = MDToolbox.rotate(model, q)
        ### loop over afmize parameters
        for iparam in 1:nparam
            param = params[iparam]
            calculated = MDToolbox.afmize(model_rotated, param)
            @save "data/calc_afms/radius_$(radius)_add_50/model_$(imodel)_iq_$(iq).bson" calculated
        end
    end
end


