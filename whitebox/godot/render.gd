extends SceneTree

var world: Node3D
var objects: Node3D
var camera: Camera3D
var scene: Dictionary
var options: Dictionary = {}
var manifest: Array = []
var asset_catalog: Dictionary = {}

func fail(message: String) -> void:
	push_error(message)
	quit(1)

func vec(value: Array) -> Vector3:
	return Vector3(float(value[0]), float(value[1]), float(value[2]))

func read_json(path: String) -> Dictionary:
	var file := FileAccess.open(path, FileAccess.READ)
	if file == null:
		fail("Cannot open " + path)
		return {}
	var parsed := JSON.new()
	if parsed.parse(file.get_as_text()) != OK or not parsed.data is Dictionary:
		fail("Expected JSON object: " + path)
		return {}
	return parsed.data

func _initialize() -> void:
	var args := OS.get_cmdline_user_args()
	for i in range(0, args.size(), 2):
		if i + 1 >= args.size():
			fail("Arguments must be --name value pairs")
			return
		options[args[i].trim_prefix("--")] = args[i + 1]
	call_deferred("run")

func material(rgb: Array) -> Material:
	if options.get("render-pass", "preview") == "depth":
		var shader := Shader.new()
		shader.code = "shader_type spatial; render_mode unshaded, fog_disabled, cull_disabled; void fragment() { float d = clamp((18.0 + VERTEX.z) / 16.0, 0.0, 1.0); ALBEDO = vec3(d); }"
		var depth_mat := ShaderMaterial.new()
		depth_mat.shader = shader
		return depth_mat
	var mat := StandardMaterial3D.new()
	mat.albedo_color = Color(float(rgb[0]), float(rgb[1]), float(rgb[2]))
	mat.roughness = 0.88
	mat.cull_mode = BaseMaterial3D.CULL_DISABLED
	return mat

func mesh_node(mesh: Mesh, pos: Vector3, mat: Material, parent: Node3D) -> MeshInstance3D:
	var node := MeshInstance3D.new()
	node.mesh = mesh
	node.material_override = mat
	node.position = pos
	parent.add_child(node)
	return node

func ball(pos: Vector3, radius: float, mat: Material, parent: Node3D) -> void:
	var mesh := SphereMesh.new()
	mesh.radius = radius
	mesh.height = radius * 2.0
	mesh.radial_segments = 16
	mesh.rings = 8
	mesh_node(mesh, pos, mat, parent)

func rod(a: Vector3, b: Vector3, radius: float, mat: Material, parent: Node3D) -> void:
	var length := a.distance_to(b)
	if length < 0.00001:
		return
	var mesh := CylinderMesh.new()
	mesh.top_radius = radius
	mesh.bottom_radius = radius
	mesh.height = length
	mesh.radial_segments = 12
	var node := mesh_node(mesh, (a + b) * 0.5, mat, parent)
	var axis := (b - a).normalized()
	var side := axis.cross(Vector3.FORWARD)
	if side.length_squared() < 0.001:
		side = axis.cross(Vector3.RIGHT)
	side = side.normalized()
	node.basis = Basis(side, axis, side.cross(axis).normalized())

func box(pos: Vector3, size: Vector3, yaw: float, mat: Material, parent: Node3D) -> void:
	var mesh := BoxMesh.new()
	mesh.size = size
	mesh_node(mesh, pos, mat, parent).rotation.y = yaw

func height_at(x: float, z: float, terrain: Dictionary) -> float:
	var seed_value := float(terrain.get("seed", scene.get("seed", 0)))
	var amplitude := float(terrain.get("height_amplitude", 0.0))
	var flatten := float(terrain.get("flatten_radius", 5.0))
	var h := amplitude * (0.5 * sin(0.31*x + seed_value*0.013) * cos(0.27*z - seed_value*0.017) + 0.25*sin(0.61*(x+z) + seed_value*0.007))
	return h * clampf((sqrt(x*x + z*z) - flatten)/3.0, 0.0, 1.0)

func terrain_point(ix: int, iz: int, n: int, size: Array, terrain: Dictionary) -> Vector3:
	var x := (float(ix)/n - 0.5) * float(size[0])
	var z := (float(iz)/n - 0.5) * float(size[1])
	return Vector3(x, height_at(x, z, terrain), z)

func build_terrain() -> void:
	if scene.has("static_geometry"):
		var path: String = scene["static_geometry"]["path"]
		if not path.is_absolute_path(): path = str(options["scene"]).get_base_dir().path_join(path)
		var file := FileAccess.open(path, FileAccess.READ)
		if file == null:
			fail("Cannot open static geometry: " + path)
			return
		var data := file.get_buffer(file.get_length()).to_float32_array()
		if data.size() == 0 or data.size() % 18 != 0:
			fail("Static geometry must contain XYZ RGB triangle vertices")
			return
		var surface := SurfaceTool.new()
		surface.begin(Mesh.PRIMITIVE_TRIANGLES)
		for i in range(0,data.size(),6):
			surface.set_color(Color(data[i+3],data[i+4],data[i+5]))
			surface.add_vertex(Vector3(data[i],data[i+1],data[i+2]))
		var mat := material([1,1,1])
		if mat is StandardMaterial3D:
			mat.vertex_color_use_as_albedo = true
			mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
		mesh_node(surface.commit(),Vector3.ZERO,mat,world)
		return
	var terrain: Dictionary = scene["terrain"]
	var size: Array = terrain["size"]
	var n := int(terrain["resolution"])
	var surface := SurfaceTool.new()
	surface.begin(Mesh.PRIMITIVE_TRIANGLES)
	for z in range(n):
		for x in range(n):
			var a := terrain_point(x, z, n, size, terrain)
			var b := terrain_point(x+1, z, n, size, terrain)
			var c := terrain_point(x, z+1, n, size, terrain)
			var d := terrain_point(x+1, z+1, n, size, terrain)
			for vertex in [a,b,c,b,d,c]:
				surface.add_vertex(vertex)
	surface.generate_normals()
	mesh_node(surface.commit(), Vector3.ZERO, material([0.36,0.46,0.39]), world)
	if options.get("render-pass", "preview") == "depth":
		return
	# A sparse ground grid makes scale and contact readable in review images.
	var grid_mat := material([0.46,0.55,0.48])
	for x in range(-int(float(size[0])/2), int(float(size[0])/2)+1, 2):
		for z in range(n):
			var z0 := (float(z)/n-0.5)*float(size[1])
			var z1 := (float(z+1)/n-0.5)*float(size[1])
			rod(Vector3(x,height_at(x,z0,terrain)+0.012,z0),Vector3(x,height_at(x,z1,terrain)+0.012,z1),0.008,grid_mat,world)
	for z in range(-int(float(size[1])/2), int(float(size[1])/2)+1, 2):
		for x in range(n):
			var x0 := (float(x)/n-0.5)*float(size[0])
			var x1 := (float(x+1)/n-0.5)*float(size[0])
			rod(Vector3(x0,height_at(x0,z,terrain)+0.012,z),Vector3(x1,height_at(x1,z,terrain)+0.012,z),0.008,grid_mat,world)

func actor(entity: Dictionary, parent: Node3D, mat: Material) -> void:
	# The dataset uses fixed parts, without synthesized joint animation.
	catalog_asset("gta_human", entity, parent, mat)

func catalog_asset(asset_id: String, entity: Dictionary, parent: Node3D, mat: Material) -> void:
	var asset: Dictionary = asset_catalog["assets"][asset_id]
	var reference_size := vec(asset["size"])
	var scale_value := vec(entity.get("size", asset["size"])) / reference_size
	if entity["kind"] == "actor":
		var posture: Dictionary = entity.get("posture_weights", {"standing":1.0})
		var height_factor := 0.0
		for name in posture: height_factor += float(posture[name])*float(asset_catalog["height_factors"][name])
		scale_value.y *= height_factor
	var origin := vec(entity["position"])
	# Existing boxes use a centre origin; actors use a foot origin.
	if entity["kind"] == "box": origin.y -= reference_size.y * scale_value.y * 0.5
	var yaw := float(entity.get("rotation_y", 0.0))
	for part in asset["parts"]:
		var surface := SurfaceTool.new()
		surface.begin(Mesh.PRIMITIVE_TRIANGLES)
		for face in asset_catalog["faces"]:
			for index in [face[0],face[1],face[2],face[0],face[2],face[3]]:
				var vertex := vec(part["vertices"][index]) * scale_value
				surface.add_vertex(vertex.rotated(Vector3.UP,yaw))
		surface.generate_normals()
		mesh_node(surface.commit(),origin,mat,parent)

func articulated_debug_actor(entity: Dictionary, parent: Node3D, mat: Material) -> void:
	var size := vec(entity.get("size", [0.6,1.8,0.4]))
	var h := size.y
	var w := size.x
	var root_pos := vec(entity["position"])
	var yaw := float(entity.get("rotation_y", 0.0))
	var defaults := {
		"pelvis":Vector3(0,h*0.51,0), "chest":Vector3(0,h*0.73,0), "head":Vector3(0,h*0.91,0),
		"left_shoulder":Vector3(-w*0.5,h*0.75,0), "right_shoulder":Vector3(w*0.5,h*0.75,0),
		"left_elbow":Vector3(-w*0.65,h*0.57,0.03), "right_elbow":Vector3(w*0.65,h*0.57,0.03),
		"left_hand":Vector3(-w*0.7,h*0.40,0.06), "right_hand":Vector3(w*0.7,h*0.40,0.06),
		"left_hip":Vector3(-w*0.23,h*0.51,0), "right_hip":Vector3(w*0.23,h*0.51,0),
		"left_knee":Vector3(-w*0.25,h*0.28,0.035), "right_knee":Vector3(w*0.25,h*0.28,0.035),
		"left_foot":Vector3(-w*0.26,h*0.045,0.08), "right_foot":Vector3(w*0.26,h*0.045,0.08)}
	var pose: Dictionary = entity.get("pose", {})
	var joints: Dictionary = {}
	for key in defaults:
		joints[key] = vec(pose[key]) if pose.has(key) else root_pos + (defaults[key] as Vector3).rotated(Vector3.UP,yaw)
	for pair in [["pelvis","chest"],["left_shoulder","right_shoulder"],["left_hip","right_hip"],["chest","head"],
		["left_shoulder","left_elbow"],["left_elbow","left_hand"],["right_shoulder","right_elbow"],["right_elbow","right_hand"],
		["left_hip","left_knee"],["left_knee","left_foot"],["right_hip","right_knee"],["right_knee","right_foot"]]:
		var radius := h*0.095 if pair[0] == "pelvis" else h*0.036
		rod(joints[pair[0]],joints[pair[1]],radius,mat,parent)
	var joint_mat := material([0.17,0.19,0.22])
	for key in joints:
		ball(joints[key], h*0.095 if key == "head" else h*0.044, mat if key == "head" else joint_mat,parent)
	for foot in ["left_foot", "right_foot"]:
		box(joints[foot], Vector3(h*0.10,h*0.088,h*0.18),yaw,joint_mat,parent)
	var forward := Vector3(sin(yaw),0,cos(yaw))
	ball(joints["head"] + forward*h*0.092,h*0.025,joint_mat,parent)

func build_entities(frame: Dictionary) -> void:
	for child in objects.get_children():
		objects.remove_child(child)
		child.queue_free()
	var overrides: Dictionary = {}
	for update in frame.get("entities", []):
		overrides[str(update["id"])] = update
	for original in scene["entities"]:
		var entity: Dictionary = original.duplicate(true)
		entity.merge(overrides.get(str(entity["id"]), {}),true)
		var mat := material(entity.get("color", [0.72,0.76,0.80]))
		var pos := vec(entity["position"])
		var size := vec(entity.get("size", [1,1,1]))
		var yaw := float(entity.get("rotation_y", 0.0))
		match entity["kind"]:
			"actor": actor(entity,objects,mat)
			"box":
				if entity.get("asset_id", "") == "gta_vehicle":
					catalog_asset("gta_vehicle",entity,objects,mat)
				else:
					box(pos,size,yaw,mat,objects)
			"staff":
				if not entity.has("weapon"):
					var delta := Vector3(0,size.y*0.5,0)
					rod(pos-delta,pos+delta,maxf(size.x*0.5,0.025),mat,objects)
			"stairs":
				var steps := int(entity.get("steps", entity.get("step_count", 6)))
				for i in range(steps):
					var step_h := size.y*float(i+1)/steps
					var local := Vector3(0,-size.y*0.5+step_h*0.5,-size.z*0.5+size.z*(i+0.5)/steps)
					box(pos+local.rotated(Vector3.UP,yaw),Vector3(size.x,step_h,size.z/steps),yaw,mat,objects)
		if entity.has("weapon"):
			var weapon: Dictionary = entity["weapon"]
			var a := vec(weapon["a"])
			var b := vec(weapon["b"])
			var weapon_mat := mat if entity["kind"] == "staff" else material([0.82,0.65,0.24])
			rod(a,b,float(weapon.get("radius",0.035)),weapon_mat,objects)
			ball(a,0.05,weapon_mat,objects)
			ball(b,0.05,weapon_mat,objects)

func set_camera(spec: Dictionary) -> void:
	var width := int(options.get("width",960))
	var height := int(options.get("height",640))
	if spec.has("intrinsics"):
		var k: Dictionary = spec["intrinsics"]
		var fx := float(k["fx"]) * width / float(k["width"])
		var fy := float(k["fy"]) * height / float(k["height"])
		root.size = Vector2i(roundi(width*fy/fx),height)
		camera.fov = rad_to_deg(2.0*atan(height/(2.0*fy)))
	else:
		root.size = Vector2i(width,height)
		camera.fov = float(spec.get("fov", 50.0))
	camera.position = vec(spec["position"])
	var target := vec(spec["target"])
	var direction := (target-camera.position).normalized()
	camera.look_at(target, vec(spec["up"]) if spec.has("up") else (Vector3.FORWARD if absf(direction.dot(Vector3.UP)) > 0.99 else Vector3.UP))
	camera.near = float(spec.get("near",0.05))

func capture(name: String, frame_index: int, time: float) -> bool:
	for i in range(3):
		await process_frame
		await RenderingServer.frame_post_draw
	var image := root.get_texture().get_image()
	if image == null or image.is_empty():
		fail("Renderer returned an empty image; do not use --headless")
		return false
	var min_luma := 1.0
	var max_luma := 0.0
	var changed := 0
	var first := image.get_pixel(0,0)
	for y in range(0,image.get_height(),8):
		for x in range(0,image.get_width(),8):
			var pixel := image.get_pixel(x,y)
			var luma := pixel.get_luminance()
			min_luma = minf(min_luma,luma)
			max_luma = maxf(max_luma,luma)
			if absf(pixel.r-first.r)+absf(pixel.g-first.g)+absf(pixel.b-first.b) > 0.08: changed += 1
	if max_luma-min_luma < 0.08 or changed < 32:
		fail("Blank or near-uniform render: " + name)
		return false
	var path: String = options["output"].path_join(name + ".png")
	image.resize(int(options.get("width",960)),int(options.get("height",640)),Image.INTERPOLATE_BILINEAR)
	if image.save_png(path) != OK:
		fail("Cannot write " + path)
		return false
	var projections: Array = []
	for point in scene.get("projection_check_points",[]):
		var pixel := camera.unproject_position(vec(point))
		projections.append([pixel.x*float(options.get("width",960))/root.size.x,pixel.y*float(options.get("height",640))/root.size.y])
	manifest.append({"file":name+".png","frame":frame_index,"time":time,"luminance_range":max_luma-min_luma,"changed_samples":changed,"projection_check_pixels":projections,"camera":{"position":[camera.position.x,camera.position.y,camera.position.z]}})
	print("CAPTURE ", path)
	return true

func run() -> void:
	if not options.has("scene") or not options.has("output"):
		fail("Required: --scene scene.json --output directory")
		return
	if DisplayServer.get_name() == "headless":
		fail("Godot headless uses dummy rendering. Use Xvfb + --display-driver x11.")
		return
	scene = read_json(options["scene"])
	if scene.is_empty(): return
	asset_catalog = read_json(get_script().resource_path.get_base_dir().path_join("../../assets/gta/catalog.json"))
	if asset_catalog.is_empty(): return
	if int(scene.get("schema_version",0)) != 1:
		fail("Unsupported scene schema_version")
		return
	DirAccess.make_dir_recursive_absolute(options["output"])
	root.size = Vector2i(int(options.get("width",960)),int(options.get("height",640)))
	world = Node3D.new()
	root.add_child(world)
	objects = Node3D.new()
	world.add_child(objects)
	var environment := WorldEnvironment.new()
	environment.environment = Environment.new()
	environment.environment.background_mode = Environment.BG_COLOR
	environment.environment.background_color = Color(0.77,0.82,0.87)
	if options.get("render-pass", "preview") == "depth":
		environment.environment.background_color = Color.BLACK
	environment.environment.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	environment.environment.ambient_light_color = Color.WHITE
	environment.environment.ambient_light_energy = 0.65
	world.add_child(environment)
	var light := DirectionalLight3D.new()
	light.rotation_degrees = Vector3(-48,-32,0)
	light.light_energy = 1.3
	light.shadow_enabled = options.get("render-pass", "preview") == "preview"
	world.add_child(light)
	camera = Camera3D.new()
	camera.far = 2000.0
	world.add_child(camera)
	camera.current = true
	build_terrain()
	build_entities({})
	var extent := maxf(float(scene["terrain"]["size"][0]),float(scene["terrain"]["size"][1]))
	var target := Vector3.ZERO
	var views: Array = ["camera"] if options.get("render-pass", "preview") == "depth" else ["overview","front","side","camera"]
	for view in views:
		var position := Vector3(extent*0.75,extent*0.8,extent*0.95)
		if view == "front": position = Vector3(0,extent*0.35,extent*1.2)
		if view == "side": position = Vector3(extent*1.2,extent*0.35,0)
		set_camera(scene["camera"] if view == "camera" else {"position":[position.x,position.y,position.z],"target":[target.x,target.y,target.z]})
		if not await capture(view,-1,0.0): return
	if options.has("timeline"):
		var timeline := read_json(options["timeline"])
		if timeline.is_empty(): return
		var frames: Array = timeline["frames"]
		var indices: Array = []
		var selection: String = options.get("frames","selected")
		if selection == "all": indices = range(frames.size())
		elif selection == "selected":
			for sample in range(9):
				var index := roundi(float(sample)*(frames.size()-1)/8.0)
				if index >= 0 and not index in indices: indices.append(index)
		else:
			for token in selection.split(","): indices.append(int(token))
		# Accumulate sparse frame updates, including skipped frames.
		var state: Dictionary = {}
		var camera_spec: Dictionary = scene["camera"]
		for i in range(frames.size()):
			for entity in frames[i].get("entities",[]):
				var id := str(entity["id"])
				if not state.has(id): state[id] = {}
				state[id].merge(entity,true)
			if frames[i].has("camera"): camera_spec = frames[i]["camera"]
			if not i in indices: continue
			build_entities({"entities":state.values()})
			set_camera(camera_spec)
			if not await capture("frame_%06d" % i,i,float(frames[i]["time"])): return
	var terrain_samples: Array = []
	for point in [[0.0,0.0],[2.0,1.0],[6.0,0.0],[-7.0,4.0],[8.0,-8.0]]:
		terrain_samples.append({"x":point[0],"z":point[1],"height":height_at(point[0],point[1],scene["terrain"])})
	var output := FileAccess.open(options["output"].path_join("manifest.json"),FileAccess.WRITE)
	output.store_string(JSON.stringify({"renderer":RenderingServer.get_video_adapter_name(),"render_pass":options.get("render-pass", "preview"),"depth_encoding":"relative grayscale clamp((18-camera_z)/16,0,1); near white; far black","images":manifest,"terrain_samples":terrain_samples},"  "))
	output.close()
	quit(0)
