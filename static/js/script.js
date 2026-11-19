/**
 * CyberShield: Cyberbullying Detection from Social Media
 * Frontend Interactions & Client-Side Logic
 */

document.addEventListener("DOMContentLoaded", () => {
    // -------------------------------------------------------------------------
    // 1. Character Counter & Max Limit Warnings
    // -------------------------------------------------------------------------
    const textarea = document.getElementById("commentText");
    const charCounter = document.getElementById("charCount");
    const charCounterContainer = document.getElementById("charCounter");
    const maxChars = 5000;

    function updateCharacterCount() {
        if (!textarea || !charCounter) return;

        const currentLength = textarea.value.length;
        charCounter.textContent = currentLength.toLocaleString();

        // Color threshold transitions
        if (currentLength >= maxChars) {
            charCounterContainer.classList.add("text-danger");
            charCounterContainer.classList.remove("text-warning", "text-secondary");
        } else if (currentLength >= maxChars * 0.9) {
            charCounterContainer.classList.add("text-warning");
            charCounterContainer.classList.remove("text-danger", "text-secondary");
        } else {
            charCounterContainer.classList.add("text-secondary");
            charCounterContainer.classList.remove("text-danger", "text-warning");
        }
    }

    if (textarea) {
        textarea.addEventListener("input", updateCharacterCount);
        // Initialize count on page load (in case of browser cache/refill)
        updateCharacterCount();
    }

    // -------------------------------------------------------------------------
    // 2. Clear Textarea Button
    // -------------------------------------------------------------------------
    const clearTextBtn = document.getElementById("clearTextBtn");
    if (clearTextBtn && textarea) {
        clearTextBtn.addEventListener("click", () => {
            textarea.value = "";
            textarea.focus();
            updateCharacterCount();
        });
    }

    // -------------------------------------------------------------------------
    // 3. Form Validation & Loading Spinner on Submit
    // -------------------------------------------------------------------------
    const analyzerForm = document.getElementById("analyzerForm");
    const analyzeBtn = document.getElementById("analyzeBtn");
    const analyzeSpinner = document.getElementById("analyzeSpinner");
    const analyzeBtnText = document.getElementById("analyzeBtnText");

    if (analyzerForm && textarea) {
        analyzerForm.addEventListener("submit", (e) => {
            const rawVal = textarea.value;
            const trimmedVal = rawVal.trim();

            // Client-side validation: Reject empty or whitespace-only inputs
            if (!trimmedVal) {
                e.preventDefault();
                alert("Please enter text before analyzing. Empty comments cannot be evaluated.");
                textarea.focus();
                return false;
            }

            // Client-side validation: Check max length limit
            if (rawVal.length > maxChars) {
                e.preventDefault();
                alert(`The input exceeds the maximum limit of ${maxChars} characters.`);
                textarea.focus();
                return false;
            }

            // Activate loading spinner and disable button to prevent double-submit
            if (analyzeBtn && analyzeSpinner && analyzeBtnText) {
                analyzeSpinner.classList.remove("d-none");
                analyzeBtnText.innerHTML = " Analyzing Neural Patterns...";
                analyzeBtn.disabled = true;
                analyzeBtn.classList.add("opacity-75");
                
                // Allow form submit to continue
                analyzerForm.submit();
            }
        });
    }

    // -------------------------------------------------------------------------
    // 4. Click-to-Fill Example Prompts
    // -------------------------------------------------------------------------
    const samplePills = document.querySelectorAll(".sample-pill");
    samplePills.forEach((pill) => {
        pill.addEventListener("click", () => {
            const sampleText = pill.getAttribute("data-sample");
            if (textarea && sampleText) {
                textarea.value = sampleText;
                updateCharacterCount();

                // Smooth scroll up to textarea
                textarea.scrollIntoView({ behavior: "smooth", block: "center" });
                textarea.focus();

                // Gentle highlight flash effect on textarea
                textarea.style.transition = "border-color 0.3s ease, box-shadow 0.3s ease";
                textarea.style.borderColor = "#6366f1";
                textarea.style.boxShadow = "0 0 0 4px rgba(99, 102, 241, 0.4)";

                setTimeout(() => {
                    textarea.style.borderColor = "";
                    textarea.style.boxShadow = "";
                }, 800);
            }
        });
    });

    // -------------------------------------------------------------------------
    // 5. Auto-Dismiss Flash Alerts after 5 seconds
    // -------------------------------------------------------------------------
    const alerts = document.querySelectorAll(".custom-alert");
    alerts.forEach((alertEl) => {
        setTimeout(() => {
            try {
                const bsAlert = bootstrap.Alert.getOrCreateInstance(alertEl);
                if (bsAlert) {
                    bsAlert.close();
                }
            } catch (err) {
                // If bootstrap not ready or alert already closed
            }
        }, 6000);
    });

    // -------------------------------------------------------------------------
    // 6. Smooth Scroll for Page Anchors
    // -------------------------------------------------------------------------
    document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
        anchor.addEventListener("click", function (e) {
            const targetId = this.getAttribute("href");
            if (targetId && targetId !== "#") {
                const targetElement = document.querySelector(targetId);
                if (targetElement) {
                    e.preventDefault();
                    targetElement.scrollIntoView({
                        behavior: "smooth"
                    });
                }
            }
        });
    });

    // -------------------------------------------------------------------------
    // 7. Screenshot OCR Upload & Preview Handler
    // -------------------------------------------------------------------------
    const screenshotFileInput = document.getElementById("screenshotFile");
    const previewWrapper = document.getElementById("screenshotPreviewWrapper");
    const previewImg = document.getElementById("screenshotPreviewImg");
    const previewFileName = document.getElementById("screenshotFileName");
    const dropzonePrompt = document.getElementById("dropzonePrompt");
    const modelSelect = document.getElementById("modelSelect");
    const screenshotModelInput = document.getElementById("screenshotModelInput");
    const textModelInput = document.getElementById("textModelInput");
    const screenshotForm = document.getElementById("screenshotForm");
    const ocrAnalyzeBtn = document.getElementById("ocrAnalyzeBtn");
    const ocrSpinner = document.getElementById("ocrSpinner");
    const ocrBtnText = document.getElementById("ocrBtnText");

    // Sync model selection
    if (modelSelect) {
        modelSelect.addEventListener("change", () => {
            if (screenshotModelInput) screenshotModelInput.value = modelSelect.value;
            if (textModelInput) textModelInput.value = modelSelect.value;
        });
    }

    // Instant Image Preview
    if (screenshotFileInput && previewWrapper && previewImg) {
        screenshotFileInput.addEventListener("change", (e) => {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = (event) => {
                    previewImg.src = event.target.result;
                    if (previewFileName) previewFileName.textContent = `Selected: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
                    previewWrapper.classList.remove("d-none");
                    if (dropzonePrompt) dropzonePrompt.classList.add("d-none");
                };
                reader.readAsDataURL(file);
            }
        });
    }

    // Screenshot form submit loading spinner
    if (screenshotForm && ocrAnalyzeBtn) {
        screenshotForm.addEventListener("submit", () => {
            ocrAnalyzeBtn.disabled = true;
            if (ocrSpinner) ocrSpinner.classList.remove("d-none");
            if (ocrBtnText) ocrBtnText.innerHTML = '<i class="fa-solid fa-spinner fa-spin me-2"></i> Performing OCR & Analyzing...';
        });
    }
});
