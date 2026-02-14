const imageUpload = document.getElementById('imageUpload');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const status = document.getElementById('status');

let session;

// 1. Initialize ONNX Session
async function init() {
    try {
        // Use WASM for compatibility, WebGL/WebGPU for speed if available
        session = await ort.InferenceSession.create('./model.onnx', { executionProviders: ['wasm'] });
        status.innerText = "Model Ready! Upload a photo.";
        imageUpload.disabled = false;
    } catch (e) {
        status.innerText = "Failed to load model: " + e.message;
    }
}

imageUpload.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const img = new Image();
    img.src = URL.createObjectURL(file);
    
    img.onload = async () => {
        // Reset canvas and draw image
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0);
        
        status.innerText = "Detecting mites...";
        await processImage(img);
    };
});

async function processImage(img) {
    // YOLOv9 typically expects 640x640
    const modelWidth = 640;
    const modelHeight = 640;

    // Create a temporary canvas to resize image for the model
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = modelWidth;
    tempCanvas.height = modelHeight;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.drawImage(img, 0, 0, modelWidth, modelHeight);
    
    const imageData = tempCtx.getImageData(0, 0, modelWidth, modelHeight);
    const input = new Float32Array(modelWidth * modelHeight * 3);

    // Pre-processing: RGB ordering and normalization (0-1)
    for (let i = 0; i < imageData.data.length / 4; i++) {
        input[i] = imageData.data[i * 4] / 255;           // R
        input[i + modelWidth * modelHeight] = imageData.data[i * 4 + 1] / 255; // G
        input[i + 2 * modelWidth * modelHeight] = imageData.data[i * 4 + 2] / 255; // B
    }

    const tensor = new ort.Tensor('float32', input, [1, 3, modelWidth, modelHeight]);
    
    // Run Inference
    const output = await session.run({ images: tensor });
    const detections = output.output0.data; // Shape [1, 5, 8400]

    renderBoxes(detections, img.width, img.height);
}

function renderBoxes(data, originalWidth, originalHeight) {
    // YOLOv9 Output: [x_center, y_center, width, height, confidence]
    // Note: This logic assumes 1 class (Varroa).
    let count = 0;
    const threshold = 0.4;

    for (let i = 0; i < 8400; i++) {
        const confidence = data[i + 4 * 8400];
        if (confidence > threshold) {
            const cx = data[i] * (originalWidth / 640);
            const cy = data[i + 8400] * (originalHeight / 640);
            const w = data[i + 2 * 8400] * (originalWidth / 640);
            const h = data[i + 3 * 8400] * (originalHeight / 640);

            ctx.strokeStyle = "red";
            ctx.lineWidth = 3;
            ctx.strokeRect(cx - w/2, cy - h/2, w, h);
            count++;
        }
    }
    status.innerText = `Done! Found ${count} mites.`;
}

init();