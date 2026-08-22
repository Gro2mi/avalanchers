import xarray as xr
import zarr


class AvalanchersOutput:
    def __init__(self, path):
        self.path = path

    @property
    def attrs(self):
        root = zarr.open_group(self.path, mode="r")
        return dict(root.attrs)

    @property
    def title(self):
        return self.attrs["title"]

    @property
    def conventions(self):
        return self.attrs["conventions"]

    @property
    def source(self):
        return self.attrs["source"]

    @property
    def avalanchers_version(self):
        return self.attrs["avalanchers_version"]

    @property
    def avalanchers_repo(self):
        return self.attrs["avalanchers_repo"]

    @property
    def avalanchers_format(self):
        return self.attrs["avalanchers_format"]

    @property
    def sites(self):
        root = zarr.open_group(self.path, mode="r")
        return list(root.group_keys())

    def list_all_scenarios(self):
        root = zarr.open_group(self.path, mode="r")
        all_scenarios = []
        for site in self.sites:
            all_scenarios.extend(list(root[site].group_keys()))
        return all_scenarios

    def site(self, name):
        return AvalancherSite(self.path, name)

    def scenario(self, site, name):
        return AvalanchersScenario(self.path, name)

    def get_site(self, name_or_hash):
        matches = []
        for site in self.sites:
            if (
                site == name_or_hash
                or site.split("_")[0] == name_or_hash
                or site.split("_")[1] == name_or_hash
            ):
                matches.append(site)
        if len(matches) == 0:
            raise ValueError(f"Site {name_or_hash} not found")
        elif len(matches) == 1:
            return AvalancherSite(self.path, matches[0])
        else:
            raise ValueError(f"Multiple sites found for {name_or_hash}: {matches}")

    def __str__(self):
        return (
            f"{self.title}\n"
            f"  Path: {self.path}\n"
            f"  Avalanchers: {self.avalanchers_version}\n"
            f"  Format: {self.avalanchers_format}\n"
            f"  {len(self.sites)} sites and {len(self.list_all_scenarios())} scenarios:\n"
            + "\n".join(
                [
                    f"    - {site.split('_')[0]}: {', '.join(self.site(site).scenario_names)}"
                    for site in self.sites
                ]
            )
        )

    def __repr__(self):
        return (
            f"AvalanchersOutput("
            f"path={self.path!r}, "
            f"version={self.avalanchers_version!r}, "
            f"format={self.avalanchers_format!r})"
        )


class AvalancherSite:
    def __init__(self, path, name):
        self.path = path
        self.name = name
        self.name_no_hash = name.split("_")[0]
        self.hash = name.split("_")[1]

    def list_scenarios(self):
        root = zarr.open_group(self.path + "/" + self.name, mode="r")
        return list(root.group_keys())

    @property
    def dem(self):
        return xr.open_dataset(
            self.path,
            engine="zarr",
            group=self.name,
            consolidated=False,
        )["dem"]

    def get_scenario(self, name):
        return AvalanchersScenario(self.path, self.name, name)

    @property
    def scenario_names(self):
        return [s.split("_")[0] for s in self.list_scenarios()]

    @property
    def scenarios(self):
        return self.list_scenarios()

    def __str__(self):
        return (
            f"Site: {self.name}\n"
            f"  Name: {self.name_no_hash}\n"
            f"  Hash: {self.hash}\n"
            f"  Scenarios: {', '.join(self.scenario_names)}"
        )

    def __repr__(self):
        return f"AvalancherSite(path={self.path!r}, name={self.name!r})"


class AvalanchersScenario:
    def __init__(self, path, site, name):
        self.path = path
        self.site = site
        self.name = name
        self.name_no_hash = name.split("_")[0]
        self.hash = name.split("_")[1]

    @property
    def attrs(self):
        group = zarr.open_group(self.path, mode="r")[self.site][self.name]
        return dict(group.attrs)

    @property
    def aspect_release_degrees(self):
        return self.attrs["aspect_release_degrees"]

    @property
    def release_volume_m3(self):
        return self.attrs["release_volume_m3"]

    @property
    def number_of_runs(self):
        return self.attrs["number_of_runs"]

    @property
    def dataset(self):
        return xr.open_dataset(
            self.path,
            engine="zarr",
            group=self.site + "/" + self.name,
            consolidated=False,
        ).isel(run=slice(0, self.number_of_runs))

    def __str__(self):
        return (
            f"Scenario: {self.name}\n"
            f"  Aspect: {self.aspect_release_degrees}°\n"
            f"  Release volume: {self.release_volume_m3} m³\n"
            f"  Runs: {self.number_of_runs}"
        )

    def __repr__(self):
        return (
            f"AvalanchersScenario("
            f"name={self.name!r}, "
            f"aspect={self.aspect_release_degrees!r}, "
            f"release_volume_m3={self.release_volume_m3!r}, "
            f"number_of_runs={self.number_of_runs!r})"
        )
