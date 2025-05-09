
def flatten(lst):
	"""
	Flatten a list of lists
	"""
	for x in lst:
		if isinstance(x, list):
			for z in flatten(x):
				yield z
		else:
			yield x

def locations(lst):
	"""
	Flatten a list of lists and get (key, value) pairs
	"""
	for i, v in enumerate(lst):
		if isinstance(v, list):
			for suffix, value in locations(v):
				yield (i,) + suffix, value
		else:
			yield (i,), v

def structural_copy(lst):
	"""
	Produce an arrangement of lists of identical shape but filled with None
	"""
	return [structural_copy(x) if isinstance(x, list) else None for x in lst]

def nested_set(lst, path, value):
	for idx in path[:-1]:
		lst = lst[idx]
	lst[path[-1]] = value

class Field:
	def __init__(self, *args, required=True):
		"""
		Attribute descriptor.

		Internally translates e.g. Field(int, float, str, [int], [float]) into:
		FieldScope(int, float, str)
		FieldScope([int])
		FieldScope([float])

		Scoping is to handle array variants.
		"""

		class FieldScope:
			def __init__(self, path):
				self.primitive = tuple()
				self.composite = tuple()
				self.path = path

			def __repr__(self):
				return "FieldScope(" + ", ".join(
					[x.__name__ for x in self.primitive] +
					[x.__name__ for x in self.composite]
				) + ")"

			def covers(self, path):
				return len(self.path) - 1 == len(path)

		scopes = { None: FieldScope( (0,) ) }

		# Types unionized here
		for k, arg in locations(args):
			index = None if len(k) == 1 else k[:-1]
			scope = scopes.get( index, FieldScope(k) )

			if arg in [int, float, str, bool, type(None)]:
				scope.primitive += (arg,)
			else:
				scope.composite += (arg,)

			scopes[index] = scope

		self.scopes = scopes
		self.required = required

class Schema:
	@classmethod
	def fields(cls):
		return dict(filter(
			lambda kv: isinstance(kv[1], Field),
			[ (k, getattr(cls, k)) for k in dir(cls)]
		))

	@classmethod
	def required_fields(cls):
		return dict(filter(
			lambda kv: isinstance(kv[1], Field) and kv[1].required,
			[ (k, getattr(cls, k)) for k in dir(cls)]
		))

	def serialize(self):
		"""
		Translate into a dict
		"""

		def flat(x):
			if isinstance(x, list):
				return [flat(z) for z in x]

			return x.serialize() if isinstance(x, Schema) else x

		return { k: flat(getattr(self, k)) for k in self.fields() }

	@classmethod
	def trial_signatures(cls, obj, schemas):
		"""
		Picks a schema to deserialize a nested schema
		"""

		present = set(obj)
		extraneous = {}
		lacking = {}

		for schema in schemas:
			pending = set(schema.required_fields())
			allowed = set(schema.fields())

			if pending - present:
				lacking[schema] = pending - present
				continue

			if present - allowed:
				extraneous[schema] = present - allowed
				continue

			return schema.deserialize(obj)

		lacking = [f"Attempted to parse as {k.__name__} but lack required attrs {v}" for k, v in lacking.items()]
		extraneous = [f"Attempted to parse as {k.__name__} but have extraneous attrs {v}" for k, v in extraneous.items()]

		raise TypeError(f"\n" + "\n".join(lacking) + "\n".join(extraneous))

	@classmethod
	def interpret_list(cls, lst, scopes):
		"""
		List patterns can be basic:
		[int, float, str, Schema] => *:int|float|str|Schema

		Also multidimensional:
		[[int]] => *.*:int

		With alternatives:
		[int], [float] => *:int, *:float

		=== we DO NOT support those below ===

		Mixed dimensionality:
		[ int, [float] ] => *:int + *.*:float
		[ int, [float], [[str]] ] => *:int + *.*:float + *.*.*:str

		Those that need demotion:
		[ int, [float], [str] ] => please rewrite as [ int, [float] ], [ int, [str] ]
		[ int, [float, [str]], [bool, [Schema]] ] => please rewrite as [ int, [float], [[str]] ], [ int, [bool], [[Schema]] ]
		"""

		inapplicable = []

		# We must:
		# - [DONE IN Field()] Build a set of scopes
		#	- May or may not cover paths to elements in array
		# - [DONE HERE] Iterate through locations in a nested map() operation
		#	- Array structure must be cloned to serve as output
		#	- Lists never encountered
		#	- Dicts handled through trial_signatures
		#	- Primitive types handled through isinstance
		for scope in scopes.values():
			clone = structural_copy(lst)

			try:
				for k, v in locations(lst):
					if not scope.covers(k):
						raise TypeError("Dimensionality mismatch")

					schemas = scope.composite
					enforced = scope.primitive

					if isinstance(v, dict) and schemas:
						v = cls.trial_signatures(v, schemas)
					else:
						if not isinstance(v, enforced):
							raise TypeError(f"array element {k}:{type(v).__name__} = {v} is not of any allowed type")

					nested_set(clone, k, v)
			except TypeError as e:
				inapplicable.append(f"Tried to apply {scope} but failed with: {e}")
				continue

			# Reached here? success
			return clone

		raise TypeError("\n".join(inapplicable))

	@classmethod
	def deserialize(cls, obj):
		pending = cls.required_fields()
		allowed = cls.fields()
		instance = cls()

		# This will not run for nested
		# Which is fine
		if set(pending) - set(obj) != set(): raise TypeError(f"Required attr missing: {set(pending) - set(obj)}")
		if set(obj) - set(allowed) != set(): raise TypeError(f"Extraneous attrs specified: {set(obj) - set(allowed)}")

		# Attribute interpretation
		# - Type enforcement for primitive types
		# - Nesting for composite types
		#	- Dicts
		#	- Arrays
		#		- Should be thought of as a kind of dict
		#			- Should be flattened
		#			- Indices are attributes
		#				- Either covered by a rule
		#				- Or not
		for k, v in obj.items():
			scopes = allowed.get(k).scopes
			scope = scopes[None]
			enforced = scope.primitive
			schemas = scope.composite

			if isinstance(v, dict) and schemas:
				v = cls.trial_signatures(v, schemas)
			elif isinstance(v, list):
				v = cls.interpret_list(v, scopes)
			else:
				if not isinstance(v, enforced):
					raise TypeError(f"{k}:{type(v).__name__} = {v} is not of any allowed type for {scope}")

			setattr(instance, k, v)

		return instance
