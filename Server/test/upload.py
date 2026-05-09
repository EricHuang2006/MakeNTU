const fs = require('fs');
const { google } = require('googleapis');

// 1. Path to your Service Account JSON key file
const KEYFILEPATH = 'path/to/service-account-key.json';

// 2. Define the scopes required (Drive scope allows full access)
const SCOPES = ['https://www.googleapis.com/auth/drive'];

// 3. Initialize Auth and Drive Service
const auth = new google.auth.GoogleAuth({
    keyFile: KEYFILEPATH,
    scopes: SCOPES,
});

const driveService = google.drive({ version: 'v3', auth });

async function uploadImage() {
    const fileMetadata = {
        'name': 'my_uploaded_image.jpg', // Name as it will appear in Drive
        // 'parents': ['16tXR0g4jbL5se5FfhPufIubPPNevpqY6'] // Optional: ID of a specific folder
    };

    const media = {
        mimeType: 'image/jpeg',
        body: fs.createReadStream('local_image_path.jpg'), // Path to local file
    };

    try {
        const response = await driveService.files.create({
            requestBody: fileMetadata,
            media: media,
            fields: 'id', // Ask for the ID in the response
        });

        console.log('File Uploaded successfully. File ID:', response.data.id);
    } catch (err) {
        console.error('Error uploading file:', err.message);
    }
}

uploadImage();:q

