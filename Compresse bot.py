#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OrcaCompress Ultimate Speed — Single-file Optimized GitHub Edition
- Max CPU threads, max concurrency, chunk sizes augmentés
- Compatible Termux / Android / Windows (ffmpeg + ffprobe)
Modes disponibles:
  A = Ultra-rapide (préserve qualité, priorité vitesse)
  B = Ultra-compressé (taille minimale, qualité réduite)
  C = Équilibré (qualité / taille / vitesse)
"""

import os
import sys
import time
import asyncio
import subprocess
import html
from pathlib import Path
from functools import wraps
from typing import Dict
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, filters
from pyrogram import Client
from pyrogram.errors import FloodWait

# ---------------- CONFIG ----------------
BOT_TOKEN = "7883530694:AAFFrPMbV_QI3rbC59cDNqNvDL0l5MevyT8"     # <-- remplace
API_ID =  22679198              # <-- remplace
API_HASH = "f5200cf837447cc1a8e0e60176fefff7"       # <-- remplace
ADMIN_IDS = [6859953213]

WORKDIR = Path("bot_tmp"); WORKDIR.mkdir(parents=True, exist_ok=True)
MAX_FILE_SIZE = 2 * 1024 * 1024 * 1024  # 2 Go
MAX_CONCURRENT = max(1, (os.cpu_count() or 4)) * 2
TARGET_RESOLUTIONS = [1080, 720, 480, 280]

FFMPEG_HWACCEL = True
PYRO_CHUNK = 10 * 1024 * 1024  # 10 MB

PRESETS = {
    "A": { "preset": "veryfast", "crf": None, "audio_copy": True, "desc": "Ultra-rapide, qualité conservée" },
    "B": { "preset": "ultrafast", "crf": 32, "audio_copy": False, "desc": "Ultra-compressé, taille minimale" },
    "C": { "preset": "veryfast", "crf": 24, "audio_copy": True, "desc": "Équilibré qualité/taille" },
}

pyro = Client("orca_ultra_pyro", api_id=API_ID, api_hash=API_HASH, bot_token=BOT_TOKEN)
compress_semaphore = asyncio.Semaphore(MAX_CONCURRENT)
user_pref_quality: Dict[int, str] = {}

# ---------------- UTILS ----------------
def esc_html(s: str) -> str:
    return html.escape(s or "")

async def run_cmd(cmd, timeout=None):
    def _run():
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False, encoding="utf-8", timeout=timeout)
            return proc.returncode, (proc.stdout or "") + (proc.stderr or "")
        except Exception as e:
            return 1, str(e)
    return await asyncio.to_thread(_run)

async def ffprobe_info(path: Path):
    cmd = ["ffprobe","-v","error","-select_streams","v:0",
           "-show_entries","stream=width,height,duration","-of","default=noprint_wrappers=1:nokey=1", str(path)]
    code, out = await run_cmd(cmd)
    if code != 0 or not out:
        return None
    try:
        w, h, d = [float(l.strip()) for l in out.strip().splitlines()[:3]]
        return {"width": int(w), "height": int(h), "duration": d}
    except:
        return None

def default_target(h:int) -> int:
    if h >= 1080: return 720
    if h >= 720: return 480
    if h >= 480: return 280
    return h

async def generate_thumbnail(video_path: Path, thumb_path: Path, at_sec: int=10):
    cmd = ["ffmpeg","-y","-ss", str(max(1, at_sec)),"-i", str(video_path),
           "-vframes","1","-vf","scale=320:-2", str(thumb_path)]
    code, _ = await run_cmd(cmd, timeout=20)
    return code == 0 and thumb_path.exists()

def ffmpeg_supports_hwaccel():
    try:
        return "mediacodec" in (subprocess.run(["ffmpeg","-hide_banner","-hwaccels"], capture_output=True, text=True).stdout or "")
    except:
        return False

# ---------------- COMPRESSION ----------------
async def ffmpeg_compress(input_path: Path, output_path: Path, target_h: int, mode_key: str):
    mode = PRESETS.get(mode_key, PRESETS["C"])
    preset, crf, audio_copy = mode["preset"], mode["crf"], mode["audio_copy"]
    hwaccel_flag = ["-hwaccel","mediacodec"] if FFMPEG_HWACCEL and ffmpeg_supports_hwaccel() else []
    threads = max(1,(os.cpu_count() or 4))
    vf = f"scale=-2:{target_h}"
    cmd = ["ffmpeg","-y"] + hwaccel_flag + ["-i", str(input_path), "-vf", vf, "-preset", preset]
    if crf is not None:
        cmd += ["-crf", str(crf)]
    cmd += ["-profile:v","high","-level","4.0","-pix_fmt","yuv420p"]
    cmd += ["-c:a","copy"] if audio_copy else ["-c:a","aac","-b:a","96k"]
    cmd += ["-threads", str(threads), "-movflags","+faststart", str(output_path)]
    start = time.time()
    code, out = await run_cmd(cmd, timeout=7200)
    return {"returncode": code, "output": out, "elapsed": time.time() - start}

async def send_video_streamable(chat_id: int, video_path: Path, info: dict, caption: str, progress_callback=None):
    thumb_path = video_path.with_suffix(".jpg")
    thumb_ok = await generate_thumbnail(video_path, thumb_path, 2)
    thumb_arg = str(thumb_path) if thumb_ok else None
    try:
        setattr(pyro,"MAX_DOWNLOAD_CHUNK_SIZE", PYRO_CHUNK)
        setattr(pyro,"MAX_UPLOAD_CHUNK_SIZE", PYRO_CHUNK)
    except: pass
    try:
        await pyro.send_video(chat_id, video=str(video_path),
                              duration=int(info.get("duration", 0)),
                              width=int(info.get("width", 0)),
                              height=int(info.get("height", 0)),
                              thumb=thumb_arg, caption=caption, supports_streaming=True,
                              progress=progress_callback)
    finally:
        try: Path(thumb_arg).unlink(missing_ok=True)
        except: pass

# ---------------- TELEGRAM HANDLERS ----------------
def admin_only(func):
    @wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE, *a, **kw):
        uid = getattr(update.effective_user, "id", None) or getattr(update.callback_query.from_user, "id", None)
        if uid not in ADMIN_IDS:
            if update.message: await update.message.reply_text("❌ Accès réservé aux admins.")
            elif update.callback_query: await update.callback_query.answer("❌ Accès refusé.", show_alert=True)
            return
        return await func(update, context, *a, **kw)
    return wrapper

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    kb = [
        [InlineKeyboardButton("A - Ultra rapide", callback_data="preset_A"),
         InlineKeyboardButton("B - Ultra compressé", callback_data="preset_B")],
        [InlineKeyboardButton("C - Équilibré", callback_data="preset_C")]
    ]
    await update.message.reply_text("👋 Envoie une vidéo MP4 et choisis un preset :", reply_markup=InlineKeyboardMarkup(kb))

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("ℹ️ Envoie vidéo MP4 → choisis preset A/B/C → compression rapide.\nModes: A=rapide, B=minimale, C=équilibré.")

async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    data = q.data or ""
    uid = q.from_user.id if q.from_user else None
    if data.startswith("preset_"):
        user_pref_quality[uid] = data.split("_")[1].upper()
        try: await q.edit_message_text(f"✅ Preset sélectionné: {user_pref_quality[uid]}. Envoie une vidéo pour l'appliquer.")
        except: pass

# ----- Gestion vidéo et compression -----
async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message; uid = msg.from_user.id
    file_obj = getattr(msg,"video",None) or getattr(msg,"document",None)
    if not file_obj: await msg.reply_text("📎 Envoie une vidéo (.mp4)"); return
    if getattr(file_obj,"file_size",0) > MAX_FILE_SIZE: await msg.reply_text("❌ Fichier > 2Go"); return
    filename = f"{msg.message_id}_{uid}.mp4"; local_path = WORKDIR/filename
    status = await msg.reply_text("⬇️ Téléchargement en cours...")

    async def progress_dl(cur, total):
        pct = (cur/total*100) if total else 0
        try: await status.edit_text(f"⬇️ Téléchargement: {pct:.1f}%")
        except: pass

    try:
        await pyro.download_media(file_obj.file_id, file_name=str(local_path), progress=progress_dl)
    except Exception as e:
        await status.edit_text(f"❌ Erreur téléchargement: {e}")
        return

    info = await ffprobe_info(local_path)
    if not info:
        await status.edit_text("❌ Analyse impossible")
        local_path.unlink(missing_ok=True)
        return

    src_h = info["height"]
    rows = [[InlineKeyboardButton(f"{h}p", callback_data=f"do_C_{h}_{filename}")] 
            for h in TARGET_RESOLUTIONS if h < src_h]
    rows.append([InlineKeyboardButton("❌ Annuler", callback_data=f"cancel_{filename}")])

    await status.edit_text(f"🎬 {file_obj.file_name or filename}\n📏 {info['width']}x{src_h}\n⏱️ {int(info['duration']//60)}m{int(info['duration']%60):02d}s\nChoisis résolution/mode.", reply_markup=InlineKeyboardMarkup(rows))

async def callback_for_compression(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer(); data = q.data or ""
    if data.startswith("do_"):
        _, mode, h_str, filename = data.split("_",3)
        try: target_h = int(h_str)
        except: await q.edit_message_text("Hauteur invalide"); return
        asyncio.create_task(queue_compress(context, q.message.chat.id, filename, target_h, mode, q))
        await q.edit_message_text(f"⏳ Compression planifiée (mode {mode})")

async def queue_compress(context: ContextTypes.DEFAULT_TYPE, chat_id: int, filename: str, target_h: int, mode_key: str, q):
    try: await q.message.reply_text("🔁 Requête placée dans la file...")
    except: pass
    async with compress_semaphore:
        await compress_and_send(context, chat_id, filename, target_h, mode_key, q)

async def compress_and_send(context: ContextTypes.DEFAULT_TYPE, chat_id: int, filename: str, target_h: int, mode_key: str, q):
    in_path = WORKDIR/filename
    out_name = f"out_{filename.split('.')[0]}_{target_h}p_{mode_key}.mp4"
    out_path = WORKDIR/out_name
    if not in_path.exists(): await q.edit_message_text("❌ Fichier introuvable"); return
    try: await q.edit_message_text(f"⚙️ Compression → {target_h}p (mode {mode_key}) en cours...")
    except: pass

    ff = await ffmpeg_compress(in_path, out_path, target_h, mode_key)
    if ff["returncode"] != 0:
        snippet = esc_html(ff["output"][-1500:])
        try: await context.bot.send_message(chat_id, text=f"❌ Erreur encodage:\n<pre>{snippet}</pre>", parse_mode=ParseMode.HTML)
        except: pass
        in_path.unlink(missing_ok=True); out_path.unlink(missing_ok=True); return

    info = await ffprobe_info(out_path)
    caption = f"{target_h}p — mode {mode_key} ({PRESETS[mode_key]['desc']})"

    async def progress_upload(cur, total):
        pct = (cur/total*100 if total else 0)
        try: await q.edit_message_text(f"📤 Upload: {pct:.1f}%")
        except: pass

    try:
        await send_video_streamable(chat_id, out_path, info or {}, caption, progress_callback=progress_upload)
    except FloodWait as e:
        wait = getattr(e, "value", getattr(e, "x", 0))
        await q.edit_message_text(f"⏳ FloodWait {wait}s"); await asyncio.sleep(wait)
        try: await send_video_streamable(chat_id, out_path, info or {}, caption)
        except: pass

    in_path.unlink(missing_ok=True); out_path.unlink(missing_ok=True)

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f"[ERROR] {context.error}")
    try: await context.bot.send_message(chat_id=update.effective_chat.id, text=f"Erreur interne: {context.error}")
    except: pass

# ---------------- MAIN ----------------
def main():
    if BOT_TOKEN.startswith("TON_") or API_ID==12345678 or API_HASH.startswith("TON_"):
        print("Configure BOT_TOKEN, API_ID et API_HASH")
        return
    try: pyro.start(); print("✅ Pyrogram démarré")
    except Exception as e: print(f"Pyrogram start failed: {e}"); return
    try: setattr(pyro,"MAX_DOWNLOAD_CHUNK_SIZE", PYRO_CHUNK); setattr(pyro,"MAX_UPLOAD_CHUNK_SIZE", PYRO_CHUNK)
    except: pass

    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CallbackQueryHandler(callback_handler, pattern="^preset_"))
    app.add_handler(CallbackQueryHandler(callback_for_compression, pattern="^do_"))
    app.add_handler(MessageHandler((filters.VIDEO | filters.Document.MimeType("video/mp4")), handle_video))
    app.add_error_handler(error_handler)

    print(f"✅ Bot prêt — max concurrency {MAX_CONCURRENT}")
    try: app.run_polling()
    finally:
        try: pyro.stop()
        except: pass

if __name__ == "__main__":
    main()