from pxr import Gf, Sdf, UsdShade


def material(stage, path, color, roughness=0.6, metallic=0.0, texture=None):
    result = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + '/Surface')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(metallic)
    if texture:
        image = UsdShade.Shader.Define(stage, path + '/Texture')
        image.CreateIdAttr('UsdUVTexture')
        image.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(str(texture))
        reader = UsdShade.Shader.Define(stage, path + '/UV')
        reader.CreateIdAttr('UsdPrimvarReader_float2')
        reader.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('st')
        image.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), 'result')
        shader.GetInput('diffuseColor').ConnectToSource(image.ConnectableAPI(), 'rgb')
    result.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    return result
