from gliner import GLiNER

model = GLiNER.from_pretrained("fastino/gliner_multi-v1")

dream = "I dreamed about walking around in the yard with no clothes "

# issue with gliner - we would need to feed it all 887 symbols and gliner takes 5-10 labels, 
labels = ["Labor", "Harpoon", "Yarmulke", "Feast", "Key", "Scold", "YMCA", "Desert", "Cadaver", "Genocide", "Garland", "Ferret", "Naked", "Hallucination", "Mahogany", "Cactus", "Jungle", "Record", "Sauce", "Passport", "Marsupial", "Teleportation", "Waiter/Waitress", "Uranium", "Pagan", "Gills", "Camper", "Paralyzed", "Balaclava", "Lanai", "Debt", "Candy", "Team", "New", "Yoyo", "Seashells", "Rare", "Backyard", "Backstage", "Paris"]

entities = model.predict_entities(dream, labels)

for entity in entities:
    print(f"{entity['text']} → {entity['label']}")