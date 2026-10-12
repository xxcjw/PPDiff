SYSTEM_ROLE = """You are an expert user behavior analyst specializing in extracting aesthetic preferences
and consumption characteristics from product reviews.
Your analysis will be used to provide personalized product presentations for different users."""


SINGLE_INTERACTION_PROMPT = """Analyze the following user review and product information to extract the user's preference characteristics across multiple dimensions. Use whatever information is available.

[Product Title]{title}
[Product Description]{description}
[User Review]{review}
[Rating]{rating}/5.0

---

Analyze each dimension below (write "Not mentioned" if the review does not address it):

1. **Style Preference**: What visual styles, design aesthetics, or appearances does the user like or dislike?
2. **Color Preference**: What colors, tones, or color palettes does the user prefer or avoid?
3. **Material & Texture**: What materials, tactile qualities, or textures does the user favor or criticize?
4. **Packaging & Presentation**: What does the user think about packaging design or product presentation?
5. **Functionality & Practicality**: What functional features or practical needs does the user value most?
6. **Quality Perception**: What is the user's attitude toward quality, craftsmanship, and durability?
7. **Emotional Experience**: What emotional tendencies and experience keywords are expressed in the review?
8. **Price Attitude**: What is the user's view on the relationship between price and value?

Use concise keywords or short phrases for each dimension."""


AGGREGATE_PROMPT = """Below are the review analyses of one user across multiple products. Aggregate them into a concise user profile.

[Review Analyses]
{interaction_analyses}

---

Synthesize from the following perspectives:

1. **Core Style Tendency**: The user's most prominent aesthetic style (2-3 keywords)
2. **Color & Texture Preference**: Preferred color directions and material qualities
3. **Quality & Price Attitude**: Overall attitude toward quality and value for money
4. **Consumption Characteristics**: The user's purchasing habits and decision patterns
5. **Profile Summary** (3-5 sentences): A concise paragraph capturing the user's overall preference characteristics, clearly conveying "what kind of product presentation this user would prefer"

Answer in English. Be precise and concise."""


STYLE_DESCRIPTION_PROMPT = """Based on the following user profile, generate a concise visual style description in English.

[User Profile]
{profile}

Requirements:
1. Output only visual-related descriptions (style, color palette, atmosphere, texture, lighting, etc.)
2. Do not mention specific product names
3. Keep within 30-50 words
4. Output in English, using comma-separated keywords/phrases

Style description:"""


TARGET_ITEM_STYLE_PROMPT = """Based on the user's style profile and the target product information, describe how this user would ideally like to see this product presented visually.

[User Style Profile]
{profile}

[Target Product Title]{title}
[Target Product Description]{description}

---

Generate a visual presentation description for this specific product by combining the user's style profile and the product information:

1. **Style Adjustment**: How the presentation style should be adjusted to match the user's preferences (background, lighting, color tone, atmosphere, props, composition, etc.)
2. **Presentation Description** : A concise description of how this product should look when presented to this user. Focus on visual elements that can be changed (background style, color palette, lighting mood, texture emphasis, overall atmosphere).

**Important**: The background and visual presentation should harmonize with the product itself. The goal is to enhance the product's appeal, not to overshadow it.

Answer in English. The maximum length should not exceed 50-60 words. Be concise, efficient, and visual."""
