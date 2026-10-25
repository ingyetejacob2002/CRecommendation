document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("recommendationForm");
  const fillDemoBtn = document.getElementById("fillDemoBtn");
  const loadProfileBtn = document.getElementById("loadProfileBtn");
  const locationSelect = document.getElementById("locationSelect");

  /* ---------- Toast ---------- */
  function toast(message, type) {
    const region = document.getElementById("toastRegion");
    if (!region) return;
    const el = document.createElement("div");
    el.className = "toast" + (type === "error" ? " error" : "");
    el.textContent = message;
    region.appendChild(el);
    setTimeout(() => {
      el.classList.add("leaving");
      setTimeout(() => el.remove(), 300);
    }, 3200);
  }

  /* ---------- Form progress (overall, per section, rail) ---------- */
  function updateProgress() {
    if (!form) return;
    const inputs = form.querySelectorAll('input[type="number"]');
    const filled = Array.prototype.filter.call(inputs, (i) => i.value !== "").length;
    const text = document.getElementById("progressText");
    const fill = document.getElementById("progressFill");
    if (text) text.textContent = filled + " of " + inputs.length + " filled";
    if (fill) fill.style.width = (filled / inputs.length) * 100 + "%";

    form.querySelectorAll(".form-section").forEach((section) => {
      const key = section.dataset.domain;
      const fields = section.querySelectorAll('input[type="number"]');
      const done = Array.prototype.filter.call(fields, (i) => i.value !== "").length;
      const count = document.querySelector('[data-count="' + key + '"]');
      const link = document.querySelector('[data-rail="' + key + '"]');
      if (count) count.textContent = done + "/" + fields.length;
      if (link) link.classList.toggle("is-done", done === fields.length);
    });
  }

  if (form) {
    form.addEventListener("input", updateProgress);
    form.addEventListener("reset", () => setTimeout(updateProgress, 0));
    updateProgress();

    /* Highlight the rail link for the section in view */
    const links = document.querySelectorAll("[data-rail]");
    if (links.length && "IntersectionObserver" in window) {
      const observer = new IntersectionObserver(
        (entries) => {
          entries.forEach((entry) => {
            if (!entry.isIntersecting) return;
            links.forEach((l) =>
              l.classList.toggle("is-active", l.dataset.rail === entry.target.dataset.domain)
            );
          });
        },
        { rootMargin: "-30% 0px -60% 0px" }
      );
      form.querySelectorAll(".form-section").forEach((s) => observer.observe(s));
    }

    /* Loading state on submit */
    form.addEventListener("submit", () => {
      const button = form.querySelector('button[type="submit"]');
      if (button) {
        button.disabled = true;
        button.textContent = "Ranking crops...";
      }
    });
  }

  function setFields(values, skip) {
    Object.entries(values).forEach(([key, value]) => {
      if (skip && skip.includes(key)) return;
      const field = document.getElementById(key);
      if (field) field.value = value;
    });
    updateProgress();
  }

  /* ---------- Sample values ---------- */
  if (fillDemoBtn) {
    fillDemoBtn.addEventListener("click", () => {
      setFields({
        latitude: 7.73,
        longitude: 8.54,
        mean_daily_precipitation_mm: 4.5,
        std_daily_precipitation_mm: 7.0,
        min_daily_precipitation_mm: 0.0,
        max_daily_precipitation_mm: 75.0,
        total_precipitation_mm: 1350.0,
        mean_annual_precipitation_mm: 1350.0,
        mean_temperature_c: 27.0,
        std_temperature_c: 2.5,
        min_temperature_c: 20.0,
        max_temperature_c: 35.0,
        mean_max_temperature_c: 32.0,
        absolute_max_temperature_c: 39.0,
        mean_min_temperature_c: 22.0,
        absolute_min_temperature_c: 15.0,
        mean_relative_humidity_percent: 65.0,
        minimum_relative_humidity_percent: 30.0,
        maximum_relative_humidity_percent: 90.0,
        mean_solar_radiation: 18.0,
        minimum_solar_radiation: 8.0,
        maximum_solar_radiation: 28.0,
        mean_wind_speed: 2.5,
        minimum_wind_speed: 0.5,
        maximum_wind_speed: 7.0,
        soil_ph: 6.2,
        soil_organic_carbon_g_kg: 12.0,
        soil_nitrogen_g_kg: 1.2,
        clay_percent: 25.0,
        sand_percent: 55.0,
        silt_percent: 20.0,
        cec_cmolc_kg: 12.0,
        bulk_density_g_cm3: 1.35,
      });
      toast("Sample values filled in.");
    });
  }

  /* ---------- Saved location profile ---------- */
  if (loadProfileBtn && locationSelect) {
    const idleLabel = loadProfileBtn.textContent;

    loadProfileBtn.addEventListener("click", async () => {
      const id = locationSelect.value;
      if (!id) {
        toast("Choose a location first.", "error");
        locationSelect.focus();
        return;
      }

      loadProfileBtn.disabled = true;
      loadProfileBtn.textContent = "Loading...";

      try {
        const response = await fetch("/api/profile/" + encodeURIComponent(id));
        const data = await response.json();
        if (!data.success) throw new Error(data.error || "Could not load the location.");

        setFields(data.profile, ["location_id", "state", "lga"]);
        toast("Location " + id + " loaded.");
      } catch (error) {
        toast(error.message, "error");
      } finally {
        loadProfileBtn.disabled = false;
        loadProfileBtn.textContent = idleLabel;
      }
    });
  }

  /* ---------- Print ---------- */
  const printBtn = document.getElementById("printBtn");
  if (printBtn) printBtn.addEventListener("click", () => window.print());

  /* ---------- Results chart ---------- */
  if (window.recommendationChartData) {
    const canvas = document.getElementById("recommendationChart");
    if (canvas && typeof Chart !== "undefined") {
      const css = getComputedStyle(document.documentElement);
      const leaf = css.getPropertyValue("--leaf").trim() || "#1e7a4c";
      const soft = "#9cc7ab";
      const ink = css.getPropertyValue("--ink").trim() || "#12261d";
      const muted = css.getPropertyValue("--muted").trim() || "#566a5e";
      const line = css.getPropertyValue("--line").trim() || "#dbe4d9";
      const values = window.recommendationChartData.values;

      Chart.defaults.font.family = css.getPropertyValue("--font-body").trim() || "system-ui";

      new Chart(canvas, {
        type: "bar",
        data: {
          labels: window.recommendationChartData.labels.map((x) => x.replaceAll("_", " ")),
          datasets: [
            {
              label: "Suitability (%)",
              data: values,
              backgroundColor: values.map((_, i) => (i === 0 ? leaf : soft)),
              borderWidth: 0,
              borderRadius: 8,
              maxBarThickness: 56,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { display: false },
            tooltip: {
              backgroundColor: ink,
              padding: 10,
              callbacks: { label: (ctx) => " " + ctx.parsed.y.toFixed(1) + "%" },
            },
          },
          scales: {
            x: { grid: { display: false }, ticks: { color: ink } },
            y: {
              beginAtZero: true,
              max: 100,
              grid: { color: line },
              border: { display: false },
              ticks: { color: muted, callback: (v) => v + "%" },
            },
          },
        },
      });
    }
  }
});
