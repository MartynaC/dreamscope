include .env 

build_container_local:
	docker build --tag=$(IMAGE):dev .

run_container_local:
	docker run -it -e PORT=8000 -p 8080:8000 $(IMAGE):dev

build_for_production:
	docker build \
		--platform linux/amd64 \
		-t $(GCP_REGION)-docker.pkg.dev/$(GCP_PROJECT)/$(ARTIFACT_REPO)/$(IMAGE):prod \
		.

run_container_prod:
	docker run -it -e PORT=8000 -p 8080:8000 $(GCP_REGION)-docker.pkg.dev/$(GCP_PROJECT)/$(ARTIFACT_REPO)/$(IMAGE):prod \

push_image_production:
	docker push $(GCP_REGION)-docker.pkg.dev/$(GCP_PROJECT)/$(ARTIFACT_REPO)/$(IMAGE):prod

deploy_to_cloud_run:
	gcloud run deploy \
		--image $(GCP_REGION)-docker.pkg.dev/$(GCP_PROJECT)/$(ARTIFACT_REPO)/$(IMAGE):prod \
		--cpu 4 \
		--timeout 3600 \
		--memory $(MEMORY) \
		--region $(GCP_REGION)
