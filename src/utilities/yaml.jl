export parse_losses

function parse_losses(losses::Vector{Dict{Any, Any}})
    real_losses = [get_loss(l) for l in losses[1]["real_losses"]]
    bool_losses = losses[2]["boolean_losses"] === nothing ? [] : [get_loss(l) for l in losses[2]["boolean_losses"]]
    categorical_losses = losses[3]["categorical_losses"] === nothing ? [] : [get_loss(l["loss"], n_classes=l["n_classes"], bl=l["bin_loss"]) for l in losses[3]["categorical_losses"]]
    ordinal_losses = losses[4]["ordinal_losses"] === nothing ? [] : [get_loss(l["loss"], n_classes=l["n_classes"]) for l in losses[4]["ordinal_losses"]]
    return real_losses, bool_losses, categorical_losses, ordinal_losses
end

function get_loss(l::String; n_classes::Int64=0, bl::String="")
    if      l == "QuadLoss"        return QuadLoss()
    elseif  l == "HuberLoss"       return HuberLoss()
    elseif  l == "HingeLoss"       return HingeLoss()
    elseif  l == "OrdinalHingeLoss" return OrdinalHingeLoss(n_classes)
    elseif  l == "OvALoss"
        bin_loss = get_loss(bl)
        return OvALoss(n_classes, bin_loss=bin_loss)
    end
    return nothing
end
