// Mosaic Mask's site: the sample sheet opens full size (click, or Esc / click again to close).
(function () {
  var sheet = document.getElementById('sheet');
  var box = document.getElementById('lightbox');
  if (!sheet || !box) return;
  var img = box.querySelector('img');
  function close() { box.hidden = true; img.removeAttribute('src'); }
  sheet.addEventListener('click', function () {
    img.src = sheet.src;
    img.alt = sheet.alt;
    box.hidden = false;
  });
  box.addEventListener('click', close);
  document.addEventListener('keydown', function (e) { if (e.key === 'Escape' && !box.hidden) close(); });
})();
