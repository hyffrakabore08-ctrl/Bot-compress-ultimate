#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OrcaCompress Ultimate Speed — single-file
Modes disponibles:
  A = Ultra-rapide (préserve qualité, priorité vitesse)
  B = Ultra-compressé (taille minimale, qualité réduite)
  C = Équilibré (qualité / taille / vitesse)
Remplace BOT_TOKEN / API_ID / API_HASH avant de lancer.
Testé pour Termux / Android / Windows (avec ffmpeg & ffprobe installés).
"""

import os
import sys
import time
import asyncio
import subprocess
import html
from pathlib import Path
from functools import wraps
from typing import Optional, Dict

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)

from pyrogram import Client
from pyrogram.errors import FloodWait

# ---------------- CONFIG ----------------
BOT_TOKEN = "7883530694:AAFFrPMbV_QI3rbC59cDNqNvDL0l5MevyT8"     # <-- remplace
API_ID = 22679198               # <-- remplace
API_HASH = "f5200cf837447cc1a8e0e60176fefff7"       # <-- remplace
ADMIN_IDS = [6859953213]        # <-- modifie si besoin

WORKDIR = Path("bot_tmp"); WORKDIR.mkdir(parents=True, exist_ok=True)
MAX_FILE_SIZE = 2 * 1024 * 1024 * 1024  # 2 Go
MAX_CONCURRENT = 2  # nombre de compressions simultanées
TARGET_RESOLUTIONS = [1080, 720, 480, 280]

# Force usage of hwaccel if True (mais vérifie si ffmpeg supporte)
FFMPEG_HWACCEL = True

# Pyrogram chunk sizes (augmenter vitesse upload/download)
PYRO_CHUNK = 3 * 1024 * 1024  # 3 MB

# ---------------- Presets (A/B/C) ----------------
# Définitions : (preset, crf, audio_copy(bool), description)
PRESETS = {
    "A": { "preset": "veryfast", "crf": None, "audio_copy": True, "desc": "Ultra-rapide, qualité conservée (remux/transcode minimal)" },
    "B": { "preset": "ultrafast", "crf": 32, "audio_copy": False, "desc": "Ultra-compressé, taille minimale (perte visible)" },
    "C": { "preset": "veryfast", "crf": 24, "audio_copy": True, "desc": "Équilibré qualité/taille" },
}

# ---------------- Clients ----------------
pyro = Client("orca_ultra_pyro", api_id=API_ID, api_hash=API_HASH, bot_token=BOT_TOKEN)

# Semaphore pour controler concurrence
compress_semaphore = asyncio.Semaphore(MAX_CONCURRENT)

# préférences en mémoire
user_pref_quality: Dict[int,int] = {}

# Tenter d'augmenter priorité (Termux / Linux)
try:
    os.nice(-10)
except Exception:
    pass

# ---------------- Utils ----------------
def esc_html(s: str) -> str:
    return html.escape(s or "")

async def run_cmd(cmd, timeout=None):
    """Run subprocess in thread, return (retcode, stdout+stderr)."""
    def _run():
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False, encoding="utf-8", timeout=timeout)
            out = (proc.stdout or "") + (proc.stderr or "")
            return proc.returncode, out
        except Exception as e:
            return 1, str(e)
    return await asyncio.to_thread(_run)

async def ffprobe_info(path: Path):
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    code, out = await run_cmd(cmd)
    if code != 0 or not out:
        return None
    try:
        lines = [l.strip() for l in out.strip().splitlines() if l.strip()]
        if len(lines) < 3:
            return None
        w, h, dur = lines[:3]
        return {"width": int(float(w)), "height": int(float(h)), "duration": float(dur)}
    except Exception:
        return None

def default_target(h: int) -> int:
    if h >= 1080: return 720
    if h >= 720: return 480
    if h >= 480: return 280
    return h

async def generate_thumbnail(video_path: Path, thumb_path: Path, at_sec: int = 10):
    """Génère une miniature (jpeg) à at_sec."""
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(max(1, at_sec)),
        "-i", str(video_path),
        "-vframes", "1",
        "-vf", "scale=320:-2",
        str(thumb_path),
    ]
    code, out = await run_cmd(cmd, timeout=20)
    return code == 0 and thumb_path.exists()

def ffmpeg_supports_hwaccel():
    """Vérifie si ffmpeg supporte mediacodec (approx)."""
    try:
        proc = subprocess.run(["ffmpeg", "-hide_banner", "-hwaccels"], capture_output=True, text=True)
        return "mediacodec" in (proc.stdout or "")
    except Exception:
        return False

# ---------------- FFmpeg compression ----------------
async def ffmpeg_compress(input_path: Path, output_path: Path, target_h: int, mode_key: str):
    """
    Compress video with chosen preset mode (A/B/C).
    Returns dict {returncode, output, elapsed}
    """
    mode = PRESETS.get(mode_key, PRESETS["C"])
    preset = mode["preset"]
    crf = mode["crf"]
    audio_copy = mode["audio_copy"]

    hwaccel_flag = []
    if FFMPEG_HWACCEL and ffmpeg_supports_hwaccel():
        # Use mediacodec hwaccel on Android if supported by ffmpeg binary
        hwaccel_flag = ["-hwaccel", "mediacodec"]

    # threads = number of CPU cores (allow ffmpeg to use multi-thread)
    threads = max(1, (os.cpu_count() or 1))

    vf = f"scale=-2:{target_h}"

    cmd = ["ffmpeg", "-y"] + hwaccel_flag + ["-i", str(input_path), "-vf", vf, "-preset", preset]
    if crf is not None:
        cmd += ["-crf", str(crf)]
    # set reasonable profile/level/pixfmt for telegram compatibility
    cmd += ["-profile:v", "high", "-level", "4.0", "-pix_fmt", "yuv420p"]
    # audio
    if audio_copy:
        cmd += ["-c:a", "copy"]
    else:
        cmd += ["-c:a", "aac", "-b:a", "96k"]
    cmd += ["-threads", str(threads), "-movflags", "+faststart", str(output_path)]

    start = time.time()
    code, out = await run_cmd(cmd, timeout=7200)
    elapsed = time.time() - start
    return {"returncode": code, "output": out, "elapsed": elapsed}

# ---------------- Send helper (always send as video) ----------------
async def send_video_streamable(chat_id: int, video_path: Path, info: dict, caption: str, progress_callback=None):
    """
    Send the file as a Telegram video message (not document).
    Generates a thumb if needed. Uses pyro.send_video to ensure streaming mode.
    progress_callback(current, total) is optional (pyrogram progress signature).
    """
    duration = int(info.get("duration", 0)) if info else None
    width = int(info.get("width", 0)) if info else None
    height = int(info.get("height", 0)) if info else None

    thumb_path = video_path.with_suffix(".jpg")
    thumb_ok = await generate_thumbnail(video_path, thumb_path, at_sec=2)
    thumb_arg = str(thumb_path) if thumb_ok else None

    # Try to set chunk sizes if possible
    try:
        setattr(pyro, "MAX_DOWNLOAD_CHUNK_SIZE", PYRO_CHUNK)
        setattr(pyro, "MAX_UPLOAD_CHUNK_SIZE", PYRO_CHUNK)
    except Exception:
        pass

    # Send video with supports_streaming True
    try:
        await pyro.send_video(
            chat_id,
            video=str(video_path),
            duration=duration,
            width=width,
            height=height,
            thumb=thumb_arg,
            caption=caption,
            supports_streaming=True,
            progress=progress_callback
        )
    finally:
        # cleanup thumb
        try:
            if thumb_arg and Path(thumb_arg).exists():
                Path(thumb_arg).unlink(missing_ok=True)
        except Exception:
            pass

# ---------------- Telegram handlers ----------------
def admin_only(func):
    @wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE, *a, **kw):
        uid = None
        try:
            if update.effective_user:
                uid = update.effective_user.id
            elif update.callback_query and update.callback_query.from_user:
                uid = update.callback_query.from_user.id
        except Exception:
            pass
        if uid is None or uid not in ADMIN_IDS:
            try:
                if update.message:
                    await update.message.reply_text("❌ Accès refusé — réservé aux admins.")
                elif update.callback_query:
                    await update.callback_query.answer("❌ Accès refusé.", show_alert=True)
            except Exception:
                pass
            return
        return await func(update, context, *a, **kw)
    return wrapper

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = "👋 Envoie une vidéo mp4. Choisis un preset ensuite : A (rapide), B (compressé), C (équilibré)."
    kb = [
        [InlineKeyboardButton("A - Ultra rapide (qualité)", callback_data="preset_A"),
         InlineKeyboardButton("B - Ultra compressé", callback_data="preset_B")],
        [InlineKeyboardButton("C - Équilibré", callback_data="preset_C")]
    ]
    await update.message.reply_text(text, reply_markup=InlineKeyboardMarkup(kb))

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        ("ℹ️ Aide rapide\n\n"
         "1) Envoie une vidéo MP4.\n"
         "2) Choisis un preset (A/B/C) via les boutons.\n"
         "3) Le bot compresse et renvoie la vidéo (toujours en vidéo, pas document).\n\n"
         "Modes:\nA = Ultra-rapide (préserve), B = Ultra-compressé, C = Équilibré.")
    )

async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    if not q: return
    await q.answer()
    data = q.data or ""
    uid = q.from_user.id if q.from_user else None

    if data.startswith("preset_"):
        preset_key = data.split("_",1)[1].upper()
        # store preset to confirm to user (not required)
        try:
            await q.edit_message_text(f"✅ Preset sélectionné: {preset_key}. Envoie une vidéo pour l'appliquer.")
        except Exception:
            pass
        # store temporarily user preset choice (simple memory)
        user_pref_quality[uid] = preset_key
        return

    # cancellation callback from earlier flows
    if data.startswith("cancel_"):
        name = data.split("_",1)[1]
        p = WORKDIR / name
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            await q.edit_message_text("❌ Opération annulée, fichier supprimé.")
        except Exception:
            pass
        return

# ---------------- Video receive ----------------
async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    user = msg.from_user
    uid = user.id
    file_obj = getattr(msg, "video", None) or getattr(msg, "document", None)
    if not file_obj:
        await msg.reply_text("📎 Envoie une vidéo (.mp4).")
        return
    if getattr(file_obj, "file_size", None) and file_obj.file_size > MAX_FILE_SIZE:
        await msg.reply_text("❌ Fichier trop volumineux (max 2 Go).")
        return

    original_filename = getattr(file_obj, "file_name", None) or f"video_{int(time.time())}.mp4"
    unique_name = f"{msg.message_id}_{uid}.mp4"
    local_path = WORKDIR / unique_name

    status = await msg.reply_text("⬇️ Téléchargement démarré...")
    last_edit = time.time()

    async def progress_dl(current, total):
        nonlocal last_edit
        try:
            pct = (current/total)*100 if total else 0.0
            if time.time() - last_edit > 2:
                await status.edit_text(f"⬇️ Téléchargement: {pct:.1f}%")
                last_edit = time.time()
        except Exception:
            pass

    # set pyro chunk sizes to speed up (best-effort)
    try:
        setattr(pyro, "MAX_DOWNLOAD_CHUNK_SIZE", PYRO_CHUNK)
        setattr(pyro, "MAX_UPLOAD_CHUNK_SIZE", PYRO_CHUNK)
    except Exception:
        pass

    # download using pyro (persistent client)
    try:
        await pyro.download_media(file_obj.file_id, file_name=str(local_path), progress=progress_dl)
        await status.edit_text("✅ Téléchargement terminé. Analyse...")
    except FloodWait as e:
        await status.edit_text(f"⏳ FloodWait {getattr(e,'value',getattr(e,'x',0))} s.")
        await asyncio.sleep(getattr(e,'value',getattr(e,'x',0)))
        return
    except Exception as e:
        await status.edit_text(f"❌ Erreur téléchargement: {e}")
        try:
            local_path.unlink(missing_ok=True)
        except Exception:
            pass
        return

    info = await ffprobe_info(local_path)
    if not info:
        await status.edit_text("❌ Impossible d'analyser la vidéo (ffprobe failed). Fichier supprimé.")
        try:
            local_path.unlink(missing_ok=True)
        except Exception:
            pass
        return

    src_h = info["height"]
    auto_pref_res = default_target(src_h)
    # build buttons for target resolutions + modes A/B/C quick
    rows = []
    for h in TARGET_RESOLUTIONS:
        if h < src_h:
            rows.append([InlineKeyboardButton(f"{h}p", callback_data=f"comp_{h}_{unique_name}")])
    # add mode toggles A/B/C (inline)
    rows.append([
        InlineKeyboardButton("A", callback_data=f"modeA_{unique_name}"),
        InlineKeyboardButton("B", callback_data=f"modeB_{unique_name}"),
        InlineKeyboardButton("C", callback_data=f"modeC_{unique_name}"),
    ])
    rows.append([InlineKeyboardButton("❌ Annuler", callback_data=f"cancel_{unique_name}")])

    msg_text = (f"🎬 {original_filename}\n📏 {info['width']}x{src_h}\n"
                f"⏱️ {int(info['duration']//60)}m{int(info['duration']%60):02d}s\n\n"
                "Choisis la résolution et le mode (A/B/C) ci-dessus.")
    await status.edit_text(msg_text, reply_markup=InlineKeyboardMarkup(rows))

# ---------------- Callbacks for compress/send ----------------
async def callback_for_compression(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    if not q: return
    await q.answer()
    data = q.data or ""
    # data may be comp_{h}_{name} or modeX_{name}
    if data.startswith("comp_"):
        _, h_str, filename = data.split("_",2)
        try:
            target_h = int(h_str)
        except:
            await q.edit_message_text("Paramètre résolution invalide.")
            return
        # default to balanced mode if not chosen
        mode_key = "C"
        # spawn compression task
        asyncio.create_task(queue_compress(context, q.message.chat.id, filename, target_h, mode_key, q))
        await q.edit_message_text("⏳ Compression planifiée (mode par défaut C).")
        return

    if data.startswith("modeA_") or data.startswith("modeB_") or data.startswith("modeC_"):
        mode_key = data[4].upper()  # A/B/C
        filename = data.split("_",1)[1]
        # ask user to pick resolution now: we will edit message to show resolution options
        # try to infer info from file
        in_path = WORKDIR / filename
        info = await ffprobe_info(in_path) if in_path.exists() else None
        if not info:
            await q.edit_message_text("Fichier introuvable ou expiré.")
            return
        src_h = info["height"]
        # buttons resolutions
        rows = []
        for h in TARGET_RESOLUTIONS:
            if h < src_h:
                rows.append([InlineKeyboardButton(f"{h}p", callback_data=f"do_{mode_key}_{h}_{filename}")])
        rows.append([InlineKeyboardButton("❌ Annuler", callback_data=f"cancel_{filename}")])
        await q.edit_message_text(f"Mode {mode_key} sélectionné. Choisis la résolution :", reply_markup=InlineKeyboardMarkup(rows))
        return

    if data.startswith("do_"):
        # do_{mode}_{h}_{filename}
        parts = data.split("_",3)
        if len(parts) != 4:
            await q.edit_message_text("Paramètres invalides.")
            return
        _, mode_key, h_str, filename = parts
        try:
            target_h = int(h_str)
        except:
            await q.edit_message_text("Hauteur invalide.")
            return
        # spawn compression task
        asyncio.create_task(queue_compress(context, q.message.chat.id, filename, target_h, mode_key, q))
        await q.edit_message_text(f"⏳ Compression planifiée (mode {mode_key}) — vérifier la file.")
        return

    if data.startswith("cancel_"):
        name = data.split("_",1)[1]
        p = WORKDIR / name
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            await q.edit_message_text("❌ Opération annulée.")
        except Exception:
            pass
        return

# ---------------- Queue and worker ----------------
async def queue_compress(context: ContextTypes.DEFAULT_TYPE, chat_id: int, filename: str, target_h: int, mode_key: str, q):
    # notify
    try:
        await q.message.reply_text("🔁 Ta requête a été placée dans la file.")
    except Exception:
        pass
    async with compress_semaphore:
        await compress_and_send(context, chat_id, filename, target_h, mode_key, q)

async def compress_and_send(context: ContextTypes.DEFAULT_TYPE, chat_id: int, filename: str, target_h: int, mode_key: str, q):
    in_path = WORKDIR / filename
    if not in_path.exists():
        try:
            await q.edit_message_text("❌ Fichier introuvable / expiré.")
        except Exception:
            pass
        return

    out_name = f"out_{filename.split('.')[0]}_{target_h}p_{mode_key}.mp4"
    out_path = WORKDIR / out_name

    try:
        await q.edit_message_text(f"⚙️ Compression → {target_h}p (mode {mode_key}) en cours...")
    except Exception:
        pass

    ff = await ffmpeg_compress(in_path, out_path, target_h, mode_key)
    if ff["returncode"] != 0:
        snippet = esc_html(ff["output"][-1500:])
        try:
            await context.bot.send_message(chat_id=chat_id, text=f"❌ Erreur durant l'encodage (code {ff['returncode']}):\n<pre>{snippet}</pre>", parse_mode=ParseMode.HTML)
        except Exception:
            pass
        try:
            in_path.unlink(missing_ok=True)
            out_path.unlink(missing_ok=True)
        except Exception:
            pass
        return

    # info for sending
    info = await ffprobe_info(out_path)
    caption = f"{target_h}p — mode {mode_key} ({PRESETS[mode_key]['desc']})"

    # upload progress callback
    last = time.time()
    async def progress_upload(current, total):
        nonlocal last
        try:
            if time.time() - last > 2:
                pct = (current/total)*100 if total else 0
                try:
                    await q.edit_message_text(f"📤 Upload: {pct:.1f}%")
                except Exception:
                    pass
                last = time.time()
        except Exception:
            pass

    try:
        await q.edit_message_text("📤 Envoi en cours (envoi en tant que vidéo)...")
    except Exception:
        pass

    send_ok = False
    try:
        await send_video_streamable(int(chat_id), out_path, info or {}, caption, progress_callback=progress_upload)
        send_ok = True
    except FloodWait as e:
        # wait and retry
        wait = getattr(e, "value", getattr(e, "x", 0))
        await q.edit_message_text(f"⏳ FloodWait {wait}s, attente...")
        await asyncio.sleep(wait)
        try:
            await send_video_streamable(int(chat_id), out_path, info or {}, caption)
            send_ok = True
        except Exception as e:
            print(f"[retry send_video] {e}")
    except Exception as e:
        print(f"[send_video error] {e}")
        try:
            await context.bot.send_message(chat_id=chat_id, text=f"❌ Erreur envoi: {e}")
        except Exception:
            pass

    # stats
    orig = in_path.stat().st_size / (1024*1024) if in_path.exists() else 0
    comp = out_path.stat().st_size / (1024*1024) if out_path.exists() else 0
    reduction = 100 * (orig - comp) / orig if orig > 0 else 0
    try:
        if send_ok:
            await q.edit_message_text(f"🎉 Compression {target_h}p ({mode_key}) terminée !\nOriginal: {orig:.2f} Mo\nCompressé: {comp:.2f} Mo\nRéduction: {reduction:.1f}%")
        else:
            await q.edit_message_text("❌ Compression terminée, mais échec à l'envoi. Vérifie les logs.")
    except Exception:
        pass

    # cleanup
    try:
        in_path.unlink(missing_ok=True)
    except Exception:
        pass
    try:
        out_path.unlink(missing_ok=True)
    except Exception:
        pass

# ---------------- Restart pyro util for admin ----------------
async def restart_pyro(context: ContextTypes.DEFAULT_TYPE):
    try:
        await pyro.stop()
    except Exception:
        pass
    await asyncio.sleep(0.5)
    try:
        await pyro.start()
    except Exception as e:
        print(f"[restart_pyro] error: {e}")

# ---------------- Error handler ----------------
async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f"[ERROR] {context.error} - type: {type(context.error)}")
    try:
        if update and update.effective_chat:
            await context.bot.send_message(chat_id=update.effective_chat.id, text=f"Une erreur interne est survenue: {context.error}")
    except Exception:
        pass

# ---------------- MAIN ----------------
def main():
    if BOT_TOKEN.startswith("TON_") or API_ID == 12345678 or API_HASH.startswith("TON_"):
        print("ERREUR: Configure BOT_TOKEN, API_ID et API_HASH dans le fichier.")
        return

    # Start Pyrogram persistent client (blocking start)
    try:
        pyro.start()
        print("✅ Pyrogram démarré (session persistante).")
    except Exception as e:
        print(f"Impossible de démarrer Pyrogram: {e}")
        return

    # Increase pyro chunk sizes (best-effort)
    try:
        setattr(pyro, "MAX_DOWNLOAD_CHUNK_SIZE", PYRO_CHUNK)
        setattr(pyro, "MAX_UPLOAD_CHUNK_SIZE", PYRO_CHUNK)
    except Exception:
        pass

    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CallbackQueryHandler(callback_handler, pattern="^preset_"))
    app.add_handler(CallbackQueryHandler(callback_for_compression, pattern="^(comp_|mode|do_|cancel_)"))
    app.add_handler(MessageHandler((filters.VIDEO | filters.Document.MimeType("video/mp4")), handle_video))

    app.add_error_handler(error_handler)

    print("✅ Bot prêt — Ultimate Speed mode.")
    try:
        app.run_polling()
    finally:
        try:
            pyro.stop()
        except Exception:
            pass

if __name__ == "__main__":
    main()