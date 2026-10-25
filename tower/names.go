package tower

// Tensor name prefixes. Champ carries and signs bnzmix.* and does not load
// them (C7). The same names match asideTensor, so a later text mix leaves
// the towers in place (D8).

const (
	VisTower = "bnzmix.vision_tower"
	AudTower = "bnzmix.audio_tower"
	VisEmbed = "bnzmix.embed_vision"
	AudEmbed = "bnzmix.embed_audio"
)

// VisSlots is the reserved vision range: 14×14 patches (D18).
// Audio tokens start at VisSlots past the text embedding.
const VisSlots = 196
