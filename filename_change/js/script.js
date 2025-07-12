    const fileInput = document.getElementById('file-input');
    const fileList = document.getElementById('file-list');
    const renameBtn = document.getElementById('rename-btn');
    const customNameInput = document.getElementById('custom-name-input');

    let files = [];
    let properties = [];

    async function loadProperties() {
        try {
            const response = await fetch('manual.txt');
            const text = await response.text();
            properties = text.split('\n').filter(p => p.trim() !== '');
        } catch (error) {
            console.error('Error loading properties:', error);
            properties = ["거래명세서", "전표", "전자세금계산서", "캡처","영수증"];
        }
    }

    fileInput.addEventListener('change', (e) => {
        const newFiles = Array.from(e.target.files);
        files.push(...newFiles);
        renderFileList();
    });

    function renderFileList() {
        fileList.innerHTML = '';
        files.forEach((file, index) => {
            const fileItem = document.createElement('div');
            fileItem.className = 'file-item';

            const fileName = document.createElement('span');
            fileName.className = 'file-name';
            fileName.textContent = file.name;

            const propertySelect = document.createElement('select');
            propertySelect.className = 'property-select';
            
            properties.forEach(prop => {
                const option = document.createElement('option');
                option.value = prop.toLowerCase().replace(/\s/g, '_');
                option.textContent = prop;
                propertySelect.appendChild(option);
            });

            fileItem.appendChild(fileName);
            fileItem.appendChild(propertySelect);
            fileList.appendChild(fileItem);
        });

        renameBtn.disabled = files.length === 0;
    }

    renameBtn.addEventListener('click', () => {
        const customName = customNameInput.value.trim();
        if (!customName) {
            alert('Please enter a file name.');
            return;
        }

        const fileItems = document.querySelectorAll('.file-item');
        fileItems.forEach((fileItem, index) => {
            const property = fileItem.querySelector('.property-select').value;
            const originalFile = files[index];
            const extension = originalFile.name.split('.').pop();
            const newName = `${customName}_${property}.${extension}`;

            const fileReader = new FileReader();
            fileReader.onload = function() {
                const blob = new Blob([this.result], { type: originalFile.type });
                const link = document.createElement('a');
                link.href = window.URL.createObjectURL(blob);
                link.download = newName;
                link.click();
            };
            fileReader.readAsArrayBuffer(originalFile);
        });
    });

    loadProperties().then(renderFileList);